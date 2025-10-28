import pandas as pd
import numpy as np
import json
from networkx import Graph
from numba import njit
import numpy as np
from tqdm import tqdm
import random
import pickle


class Node2Vec:
    def __init__(
        self,
        nodes_file,
        edges_file,
        num_walks,
        length_walk,
        num_negative_samples,
        window_size,
        p,
        q,
        alias_path=None,
        alias_save_path=None,
    ) -> None:
        self.graph = Graph()

        print("Buliding graph")
        nodes = pd.read_csv(nodes_file, names=["node"])["node"].tolist()
        edges = pd.read_csv(edges_file, names=["node1", "node2"]).values.tolist()
        edges2 = [(edge[1], edge[0]) for edge in edges]
        edges = edges + edges2
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        self.num_walks = num_walks
        self.length_walk = length_walk
        self.num_negative_samples = num_negative_samples
        self.window_size = window_size
        self.q = q
        self.p = p
        self.alias_all = None
        if alias_path:
            print(f"Loading alias table from {alias_path}")
            with open(alias_path, "rb") as f:
                self.alias_all = pickle.load(f)
        else:
            self._get_all_alias()
            if alias_save_path:
                print(f"Saving alias table in {alias_save_path}")
                with open(alias_save_path, "wb") as f:
                    pickle.dump(self.alias_all, f)

    def random_walk(
        self,
    ):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in tqdm(range(self.num_walks), desc="Random Walking"):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_random_walk_once(start_node=node)
                walks.append(walk)
        return walks

    def _node2vec_random_walk_once(
        self,
        start_node,
    ):
        walk = [start_node]
        # 第二个点随机均匀采样一个
        adj_list = list(self.graph.neighbors(start_node))
        idx = np.random.randint(0, len(adj_list))
        node = adj_list[idx]
        walk.append(node)
        while len(walk) < self.length_walk:
            current_node = walk[-1]
            pre_node = walk[-2]
            # current_node_adj = list(self.graph.neighbors(curren_node))
            current_node_adj, probs, alias = self.alias_all[(pre_node, current_node)]
            idx = alias_sample(prob=probs, alias=alias)
            node = current_node_adj[idx]
            walk.append(node)
        return walk

    def _get_all_alias(
        self,
    ):

        nodes = list(self.graph.nodes())
        alias_all = dict()
        for edge in tqdm(self.graph.edges(), "Preparing alias tables."):
            pre = edge[0]
            cur = edge[1]
            val = []
            cur_adj = list(self.graph.neighbors(cur))
            val.append(cur_adj)
            prob, alias = self._get_trans_prob_and_alias(pre, cur_adj)
            val.extend([prob, alias])
            ### val[0] 邻居 val[1] prob val[2] alias
            alias_all[(pre, cur)] = val

            pre = edge[1]
            cur = edge[0]
            val = []
            cur_adj = list(self.graph.neighbors(cur))
            val.append(cur_adj)
            prob, alias = self._get_trans_prob_and_alias(pre, cur_adj)
            val.extend([prob, alias])
            ### val[0] 邻居 val[1] prob val[2] alias
            alias_all[(pre, cur)] = val
        self.alias_all = alias_all

    def _get_trans_prob_and_alias(self, pred_node, current_node_adj):
        pre_node_adj = list(self.graph.neighbors(pred_node))
        # 构造概率的时候同时构造别名采样表
        probs = np.zeros(len(current_node_adj))
        pre_node_adj_set = set(pre_node_adj)
        for idx, neighbor in enumerate(current_node_adj):
            if neighbor == pred_node:  # d_tx = 0
                probs[idx] = 1 / self.p
            elif neighbor in pre_node_adj_set:  # d_tx = 1
                probs[idx] = 1
            else:
                probs[idx] = 1 / self.q
        probs = probs / probs.sum()
        probs, alias = create_alias_table(probs.tolist())
        return probs, alias

    def neg_sampling(self, pos_nodes, num_samples):
        return negative_sampling(
            pos_nodes=pos_nodes,
            nodes=list(self.graph.nodes()),
            num_negative_samples=num_samples,
        )


@njit
def negative_sampling(
    pos_nodes,
    nodes,
    num_negative_samples,
):
    neg_nodes = [0] * num_negative_samples
    cnt = 0
    while cnt < num_negative_samples:
        idx = np.random.randint(0, len(nodes))
        node = nodes[idx]
        if node not in pos_nodes:
            neg_nodes[cnt] = node
            cnt += 1
    return neg_nodes


@njit
def create_alias_table(probabilities):
    n = len(probabilities)
    prob = np.zeros(n)
    alias = np.zeros(n, dtype=np.int64)
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
