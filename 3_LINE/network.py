import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from numba import njit


@njit
def one_sample(sampling_table: list, num_sample: int, filter: list):
    res = [0] * num_sample
    cnt = 0
    while cnt < num_sample:
        idx = np.random.randint(len(sampling_table))
        s = sampling_table[idx]
        if s not in filter:
            res[cnt] = s
            cnt += 1
    return res


class NetworkUnDir:
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        neg_sample_num: int,
        edge_weight_by_degree: bool,
    ) -> None:
        self.edge_weight_by_degree = edge_weight_by_degree
        self.neg_sample_num = neg_sample_num
        self.G = nx.Graph()
        nodes = (pd.read_csv(nodes_file, names=["node"])["node"] - 1).tolist()
        nodes.append(max(nodes) + 1)  ## 加一个绝对没有边的节点
        edges = (pd.read_csv(edges_file, names=["node1", "node2"]) - 1).values.tolist()
        print("Loading nodes and edges data.")
        self.sampling_table_size = len(nodes) * 20
        print("Buliding graph.")
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)
        self.nodes, self.nodes_weights = self._get_nodes_weights()
        self.edges, self.edges_weights = self._get_edges_weights()
        # self.adj_list = self._build_adj_list()
        self.node_alias_prob, self.node_alias = self._get_node_prob_alias()
        self.edge_alias_prob, self.edge_alias = self._get_edge_prob_alias()

    def _build_adj_list(
        self,
    ):
        adj_list = dict()
        for i, node in tqdm(enumerate(self.nodes), desc="Building adjacency list"):
            adj_list[node] = list(self.G.neighbors(node)) + [node]
        return adj_list

    def get_neg_adj_list(
        self,
    ):
        prob, alias = create_alias_table(self.nodes_weights)
        neg_adj_list = dict()
        num_neg = 100
        for node in tqdm(self.nodes):
            node_neighbors = list(self.G.neighbors(node)) + [node]
            neg_target_nodes = []
            if len(node_neighbors) > 0:
                while len(neg_target_nodes) < 100:
                    s = alias_sample(prob, alias)
                    if s not in node_neighbors:
                        neg_target_nodes.append(s)
                neg_adj_list[node] = neg_target_nodes
        return neg_adj_list

    def _get_nodes_weights(
        self,
    ):
        # 根据节点的度给节点赋予权重
        # 论文中是跟据节点的出度，由于是无向图，所以直接用节点的度
        print("Get nodes weights.")
        nodes, degrees = zip(*self.G.degree())
        nodes_weights = np.array(degrees) ** 0.75
        nodes_weights = nodes_weights / nodes_weights.sum()
        return nodes, nodes_weights

    def _get_edges_weights(
        self,
    ):
        # 给边赋予权重，无权边，采取两种方式，一种是均匀采样，一种是根据source节点的度
        print("Get edge weights.")
        if self.edge_weight_by_degree:
            for u, v in self.G.edges():
                degree_u = self.G.degree[u]
                self.G[u][v]["weight"] = degree_u
        else:
            for u, v in self.G.edges():
                self.G[u][v]["weight"] = 1
        edges_and_weight = [
            (u, v, data["weight"]) for u, v, data in self.G.edges(data=True)
        ]
        edges_weights = [s[2] for s in edges_and_weight]
        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights / edges_weights.sum()
        edges = [(s[0], s[1]) for s in edges_and_weight]
        return edges, edges_weights

    def _get_node_prob_alias(
        self,
    ):
        prob, alias = create_alias_table(np.array(self.nodes_weights, dtype=np.float64))
        return prob, alias

    def _get_edge_prob_alias(
        self,
    ):
        prob, alias = create_alias_table(np.array(self.edges_weights, dtype=np.float64))
        return prob, alias

    def _build_sampling_table(
        self,
    ):
        edge_sampling_table = []
        for index, weight in enumerate(self.edges_weights):
            edge_sampling_table.extend([index] * int(self.sampling_table_size * weight))
        node_sampling_table = []
        for index, weight in enumerate(self.nodes_weights):
            node_sampling_table.extend(
                [self.nodes[index]] * int(self.sampling_table_size * weight)
            )
        return node_sampling_table, edge_sampling_table

    def sample(
        self,
    ):
        # 首先进行边采样
        edge_pos_index = alias_sample(self.edge_alias_prob, self.edge_alias)
        # 获取源节点和目标节点
        source_node, target_node = self.edges[edge_pos_index]
        # 进行节点负采样，采样生成的节点不能是源节点和目标节点的邻居，如果含有这些节点需要重新进行采样
        neighbors = list(self.G.neighbors(source_node)) + [source_node]
        neg_target_nodes = []
        cnt = 0
        while len(neg_target_nodes) < self.neg_sample_num and cnt < 1000:
            idx = alias_sample(self.node_alias_prob, self.node_alias)
            s = self.nodes[idx]
            if s not in neighbors:
                neg_target_nodes.append(s)
            cnt += 1
        if cnt == 1000:  # 采样1000次没有把负样本采满，用没有边的节点补充
            neg_target_nodes = neg_target_nodes + [self.nodes[-1]] * (
                self.neg_sample_num - len(neg_target_nodes)
            )
        sample_nodes = [target_node] + neg_target_nodes
        sample_labels = [1] + [0] * self.neg_sample_num
        return {
            "source_node": source_node,
            "sample_nodes": sample_nodes,
            "sample_labels": sample_labels,
        }


# @njit
def create_alias_table(probabilities: np.ndarray):
    n = len(probabilities)
    prob = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int32)
    # 步骤 1：将概率标准化到 [0, 1] 区间并乘以 n
    scaled_prob = probabilities * n
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


# @njit
def alias_sample(prob, alias):
    n = len(prob)
    i = np.random.randint(0, n)  # 随机选择一个桶
    r = np.random.rand()  # 生成 [0,1) 区间的随机数

    # 根据概率表决定是选择 i 还是 alias[i]
    if r < prob[i]:
        return i
    else:
        return alias[i]


class NetworkUnDir2:
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        neg_sample_num: int,
        edge_weight_by_degree: bool,
    ) -> None:
        self.edge_weight_by_degree = edge_weight_by_degree
        self.neg_sample_num = neg_sample_num
        self.G = nx.Graph()
        nodes = (pd.read_csv(nodes_file, names=["node"])["node"] - 1).tolist()
        nodes.append(max(nodes) + 1)  ## 加一个绝对没有边的节点
        edges = (pd.read_csv(edges_file, names=["node1", "node2"]) - 1).values.tolist()
        print("Loading nodes and edges data.")
        self.sampling_table_size = len(nodes) * 20
        print("Buliding graph.")
        # self.G.add_nodes_from(nodes)
        # self.G.add_edges_from(edges)
        self.nodes, self.nodes_weights = self._get_nodes_weights()
        self.edges, self.edges_weights = self._get_edges_weights()
        # self.adj_list = self._build_adj_list()
        self.node_alias_prob, self.node_alias = self._get_node_prob_alias()
        self.edge_alias_prob, self.edge_alias = self._get_edge_prob_alias()

    def _build_adj_list(
        self,
    ):
        adj_list = dict()
        for i, node in tqdm(enumerate(self.nodes), desc="Building adjacency list"):
            adj_list[node] = list(self.G.neighbors(node)) + [node]
        return adj_list

    def get_neg_adj_list(
        self,
    ):
        prob, alias = create_alias_table(self.nodes_weights)
        neg_adj_list = dict()
        num_neg = 100
        for node in tqdm(self.nodes):
            node_neighbors = list(self.G.neighbors(node)) + [node]
            neg_target_nodes = []
            if len(node_neighbors) > 0:
                while len(neg_target_nodes) < 100:
                    s = alias_sample(prob, alias)
                    if s not in node_neighbors:
                        neg_target_nodes.append(s)
                neg_adj_list[node] = neg_target_nodes
        return neg_adj_list

    def _get_nodes_weights(
        self,
    ):
        # 根据节点的度给节点赋予权重
        # 论文中是跟据节点的出度，由于是无向图，所以直接用节点的度
        print("Get nodes weights.")
        nodes, degrees = zip(*self.G.degree())
        nodes_weights = np.array(degrees) ** 0.75
        nodes_weights = nodes_weights / nodes_weights.sum()
        return nodes, nodes_weights

    def _get_edges_weights(
        self,
    ):
        # 给边赋予权重，无权边，采取两种方式，一种是均匀采样，一种是根据source节点的度
        print("Get edge weights.")
        if self.edge_weight_by_degree:
            for u, v in self.G.edges():
                degree_u = self.G.degree[u]
                self.G[u][v]["weight"] = degree_u
        else:
            for u, v in self.G.edges():
                self.G[u][v]["weight"] = 1
        edges_and_weight = [
            (u, v, data["weight"]) for u, v, data in self.G.edges(data=True)
        ]
        edges_weights = [s[2] for s in edges_and_weight]
        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights / edges_weights.sum()
        edges = [(s[0], s[1]) for s in edges_and_weight]
        return edges, edges_weights

    def _get_node_prob_alias(
        self,
    ):
        prob, alias = create_alias_table(np.array(self.nodes_weights, dtype=np.float64))
        return prob, alias

    def _get_edge_prob_alias(
        self,
    ):
        prob, alias = create_alias_table(np.array(self.edges_weights, dtype=np.float64))
        return prob, alias

    def _build_sampling_table(
        self,
    ):
        edge_sampling_table = []
        for index, weight in enumerate(self.edges_weights):
            edge_sampling_table.extend([index] * int(self.sampling_table_size * weight))
        node_sampling_table = []
        for index, weight in enumerate(self.nodes_weights):
            node_sampling_table.extend(
                [self.nodes[index]] * int(self.sampling_table_size * weight)
            )
        return node_sampling_table, edge_sampling_table

    def sample(
        self,
    ):
        # 首先进行边采样
        edge_pos_index = alias_sample(self.edge_alias_prob, self.edge_alias)
        # 获取源节点和目标节点
        source_node, target_node = self.edges[edge_pos_index]
        # 进行节点负采样，采样生成的节点不能是源节点和目标节点的邻居，如果含有这些节点需要重新进行采样
        neighbors = list(self.G.neighbors(source_node)) + [source_node]
        neg_target_nodes = []
        cnt = 0
        while len(neg_target_nodes) < self.neg_sample_num and cnt < 1000:
            idx = alias_sample(self.node_alias_prob, self.node_alias)
            s = self.nodes[idx]
            if s not in neighbors:
                neg_target_nodes.append(s)
            cnt += 1
        if cnt == 1000:  # 采样1000次没有把负样本采满，用没有边的节点补充
            neg_target_nodes = neg_target_nodes + [self.nodes[-1]] * (
                self.neg_sample_num - len(neg_target_nodes)
            )
        sample_nodes = [target_node] + neg_target_nodes
        sample_labels = [1] + [0] * self.neg_sample_num
        return {
            "source_node": source_node,
            "sample_nodes": sample_nodes,
            "sample_labels": sample_labels,
        }
