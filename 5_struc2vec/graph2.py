import networkx as nx
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from tqdm import tqdm
import random
from numba import njit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class MultiLayerWeightedGraph:
    def __init__(
        self,
        base_graph: nx.Graph,
        k,
    ) -> None:
        super(MultiLayerWeightedGraph, self).__init__()
        self.k = k
        self.base_graph = base_graph
        self.graph_layers: list[nx.Graph] = [nx.Graph() for _ in range(k + 1)]
        self.nodes = list(self.base_graph.nodes())
        self.edges = list(self.base_graph.edges())
        self.node2index = {self.nodes[i]: i for i in range(len(self.nodes))}
        self.node_degrees = [self.base_graph.degree(node) for node in self.nodes]

        self.get_all_k_order_sequence()
        self.build_multilayer_graph()
        self.build_alias_tables()

    def get_k_hop_neighbors(self, graph, node, k):
        # 获取k跳邻居
        k_hop_neighbors = [
            n
            for n in nx.single_source_shortest_path_length(graph, node, cutoff=k)
            if nx.shortest_path_length(graph, node, n) == k
        ]
        return k_hop_neighbors
        # return list(nx.ego_graph(graph, node, radius=k).nodes)

    def get_ordered_degree_sequence(self, node_list):
        ordered_degree_seq = sorted(
            self.node_degrees[self.node2index[node]] for node in node_list
        )
        return ordered_degree_seq

    def get_all_k_order_sequence(self):
        all_k_degree_order_sequence = []
        for k in tqdm(range(0, self.k + 1), "Get all seq"):
            k_degree_order_seq = []
            for node in self.nodes:
                k_hop_neighbors = self.get_k_hop_neighbors(self.base_graph, node, k)
                seq = self.get_ordered_degree_sequence(k_hop_neighbors)
                k_degree_order_seq.append(seq)
            all_k_degree_order_sequence.append(k_degree_order_seq)
        self.all_k_order_sequence = all_k_degree_order_sequence

    def compute_dtw_distance(self, seq1, seq2):
        def dist_func(a, b):
            return max(a, b) / min(a, b) - 1

        distance, _ = fastdtw(seq1, seq2, dist=dist_func)
        return distance

    def compute_dtw_distance_opt1(self, seq1, seq2):
        def dist_func(a, b):
            return (max(a[0], b[0]) / min(a[0], b[0]) - 1) * max(a[1], b[1])

        count_seq1 = list(dict(Counter(seq1)).items())
        count_seq2 = list(dict(Counter(seq2)).items())
        distance, _ = fastdtw(count_seq1, count_seq2, dist=dist_func)
        return distance

    def build_multilayer_graph(
        self,
    ):
        # 构造每一层的图
        all_k_order_sequence = self.all_k_order_sequence.copy()
        for i in tqdm(range(0, self.k + 1), desc="Building multilayer graph"):
            self.graph_layers[i].add_nodes_from(self.nodes)
            processed_edges = set()
            if i > 0:
                pre_layer_graph = self.graph_layers[i - 1].copy()

            def process_edge(node1, node2):
                if (node1, node2) not in processed_edges:
                    seq1 = all_k_order_sequence[i][self.node2index[node1]]
                    seq2 = all_k_order_sequence[i][self.node2index[node2]]
                    if seq1 and seq2:
                        distance = self.compute_dtw_distance(seq1, seq2)
                        if i > 0:
                            distance += pre_layer_graph[node1][node2]["weight"]
                        self.graph_layers[i].add_edge(node1, node2, weight=distance)
                        processed_edges.add((node1, node2))
                        processed_edges.add((node2, node1))
                    return node1, node2, distance

            weight_results = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for node1 in self.nodes:
                    for node2 in self.nodes:
                        if node1 != node2:
                            futures.append(executor.submit(process_edge, node1, node2))
                for future in futures:
                    result = future.result()
                    if result:
                        weight_results.append(result)
            for node1, node2, weight in weight_results:
                self.graph_layers[i][node1][node2]["weight"] = weight
        for i in range(0, self.k + 1):
            for u, v, d in self.graph_layers[i].edges(data=True):
                d["weight"] = np.exp(-d["weight"])

        cross_layer_weight = np.zeros([self.k + 1, len(self.nodes), 2])  #  [k, n, 2]
        # [[[down_layer_weight,up_layer_weight],....],]
        # 计算层与层之间节点边的权重
        ## 先处理第0层的，第0层只能往上走
        cross_layer_weight[0, :, 0] = 0  #  由于第0层不能往下走，所有往下走的权重为0
        cross_layer_weight[0, :, 1] = 1
        ## 在处理第k层的，第k层只能往下走
        cross_layer_weight[self.k, :, 0] = 1
        cross_layer_weight[self.k, :, 1] = 0
        ## 处理后面几层的
        for k in tqdm(range(1, self.k)):
            cross_layer_weight[k, :, 0] = 1
            edge_weights = nx.get_edge_attributes(self.graph_layers[k], "weight")
            edge_weights = [w[1] for w in edge_weights.items()]
            average_weight = np.array(edge_weights).mean()
            for i in range(len(self.nodes)):
                u = self.nodes[i]
                edges = self.graph_layers[k].edges(u, data=True)
                count = sum(1 for _, _, d in edges if d["weight"] > average_weight)
                up_weight = np.log(count + np.e)
                cross_layer_weight[k, i, 1] = up_weight
        # 根据每个节点的层转移权重计算层转移概率
        cross_layer_prob = np.zeros_like(cross_layer_weight)
        for k in range(self.k + 1):
            for i in range(len(self.nodes)):
                total_weight = cross_layer_weight[k, i, :].sum(keepdims=True)
                cross_layer_prob[k, i, :] = cross_layer_weight[k, i, :] / total_weight
        self.cross_layer_prob = cross_layer_prob  # [k,n,2]

    def build_alias_tables(
        self,
    ):
        # one node alias table contains [neighbor nodes, prob, alias_table]
        alias_tables_all = []
        for k in tqdm(range(0, self.k + 1), desc="Building alias tables"):
            k_layer_tables = []
            for node in self.nodes:
                node_edges = list(self.graph_layers[k].edges(node, data=True))
                if node_edges:
                    neighbors_nodes = [s[1] for s in node_edges]
                    weights = np.array([s[2]["weight"] for s in node_edges])
                    prob = (weights / weights.sum()).tolist()
                    prob, alias = create_alias_table(prob)
                    k_layer_tables.append([neighbors_nodes, prob, alias])
                else:
                    k_layer_tables.append([])
            alias_tables_all.append(k_layer_tables)
        self.alias_tables_all = alias_tables_all  # [k+1,n,3]

    def random_walks(self, num_walks, length_walk, p):
        walks_all = []
        current_layer = 0
        total_walks = num_walks * len(self.nodes)
        pbar = tqdm(total=total_walks, desc="Random walking")
        nodes = self.nodes.copy()
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                current_node = node
                walks = [current_node]
                while len(walks) < length_walk:
                    current_node_idx = self.node2index[current_node]
                    # 选择是否变换层数
                    if np.random.rand() < p:
                        # 停留这一层
                        flag, next_node = self.sample_a_neighbor(
                            layer=current_layer, node_idx=current_node_idx
                        )
                    else:
                        # 变换层数
                        ## 选择是往上还是往下
                        s = np.random.rand()
                        up_prob = self.cross_layer_prob[current_layer][
                            current_node_idx
                        ][1]
                        if s < up_prob:
                            # 往上走
                            current_layer += 1
                            assert current_layer <= self.k
                        else:
                            current_layer -= 1
                            assert current_layer >= 0
                        flag, next_node = self.sample_a_neighbor(
                            layer=current_layer, node_idx=current_node_idx
                        )
                    if flag:
                        walks.append(next_node)
                        current_node = next_node
                        pbar.update(1)
                walks_all.append(walks)
        return walks_all

    def sample_a_neighbor(self, layer, node_idx):
        alias_data = self.alias_tables_all[layer]
        if alias_data[node_idx]:
            neighbor_nodes, prob, alias = alias_data[node_idx]
            idx = alias_sample(prob, alias)
            new_node = neighbor_nodes[idx]
            return True, new_node
        else:
            return False, None


@njit
def create_alias_table(probabilities):
    n = len(probabilities)
    prob = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int32)
    # 步骤 1：将概率标准化到 [0, 1] 区间并乘以 n
    scaled_prob = np.array(probabilities) * n
    # 步骤 2：划分大概率和小概率组
    small = []
    large = []
    for i, p in enumerate(scaled_prob):
        if p < 1:
            small.append(i)
        else:
            large.append(i)
    # 步骤 3：构建别名表
    while small and large:
        small_idx = small.pop()
        large_idx = large.pop()
        prob[small_idx] = scaled_prob[small_idx]
        alias[small_idx] = large_idx
        # 调整 large_idx 的概率
        scaled_prob[large_idx] = scaled_prob[large_idx] + scaled_prob[small_idx] - 1
        # 如果调整后的 large_idx 小于 1，将其放入 small 组
        if scaled_prob[large_idx] < 1:
            small.append(large_idx)
        else:
            large.append(large_idx)
    # 处理剩余的元素
    while large:
        large_idx = large.pop()
        prob[large_idx] = 1
    while small:
        small_idx = small.pop()
        prob[small_idx] = 1
    return prob, alias


@njit
def alias_sample(prob, alias):
    n = len(prob)
    i = np.random.randint(0, n)  # 随机选择一个桶
    r = np.random.rand()  # 生成 [0,1) 区间的随机数

    # 根据概率表决定是选择 i 还是 alias[i]
    if r < prob[i]:
        return i
    else:
        return alias[i]


if __name__ == "__main__":
    # 创建两个空手道俱乐部图
    G1 = nx.karate_club_graph()
    G2 = nx.karate_club_graph()

    # 将两个图联合起来
    G = nx.disjoint_union(G1, G2)

    # 在第0个节点处用一条边连接两个图
    G.add_edge(0, len(G1))

    # 创建多层加权图对象
    k = 3  # 假设k为3
    multi_layer_graph = MultiLayerWeightedGraph(base_graph=G, k=k)

    # 构建多层图
    multi_layer_graph.build_multilayer_graph()

    # 构建别名表
    multi_layer_graph.build_alias_tables()

    # 进行随机游走
    num_walks = 10
    length_walk = 5
    p = 0.5
    walks = multi_layer_graph.random_walks(num_walks, length_walk, p)

    # 打印随机游走结果
    for walk in walks:
        print(walk)
