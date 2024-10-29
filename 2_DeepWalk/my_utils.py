import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import pickle
import json
from pathlib import Path


def one_node_random_walk(
    adj_list: list,
    begin_node: int,
    t: int,
):
    """一次随机游走

    Args:
        adj_list (list): 邻接表
        begin_node (int): 最开始的节点
        t (int): 游走的最长长度

    Returns:
        list: 一次随机游走得到的序列
    """
    sequence = []
    current_node = begin_node
    sequence.append(current_node)
    while len(sequence) < t:
        adj_nodes = adj_list[current_node]
        next_index = np.random.choice(np.arange(len(adj_nodes)))
        current_node = adj_nodes[next_index]
        sequence.append(current_node)
    return sequence


class SkipGramHierarchicalSoftmaxDataset(Dataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        window_size: int,
        walks_num: int,
        t: int,
        bias: int,
        save_path: str = None,
        load_path: str = None,
        load_path_list: list = None,
    ) -> None:
        super().__init__()
        self.nodes = pd.read_csv(nodes_file, names=["node"])
        self.edges = pd.read_csv(edges_file, names=["node1", "node2"])
        self.walks_num = walks_num
        self.window_size = window_size
        self.t = t
        self.tree_height = int(np.ceil(np.log2(self.nodes.shape[0])))
        self.adj_list = None
        if load_path_list != None:
            self.data = []
            for path in tqdm(load_path_list, desc="Loading and process data"):
                with open(path, "rb") as f:
                    d = pickle.load(f)
                for sequence in d:
                    self.data += self._gain_data_from_seq(sequence)
        elif load_path != None:
            with open(load_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.adj_list = self._get_adj_list()
            self.data = self._data_process()
        self.bias = bias
        if save_path != None:
            with open(save_path, "wb") as f:
                pickle.dump(self.data, f)

    def _data_process(
        self,
    ):
        data = []
        for gamma in tqdm(range(self.walks_num), desc="Random Walking"):
            nodes = self.nodes["node"].tolist().copy()
            random.shuffle(nodes)
            for node in nodes:
                sequence = one_node_random_walk(
                    self.adj_list, begin_node=node, t=self.t
                )
                data += self._gain_data_from_seq(sequence=sequence)
        return data

    def _gain_data_from_seq(self, sequence):

        # 保证两个窗口加上中间节点的长度小于序列的长度
        assert (self.window_size * 2 + 1) < len(sequence)
        data = []
        for i in range(self.window_size, len(sequence) - self.window_size):
            data.append(sequence[i - self.window_size : i + self.window_size + 1])
        return data

    def _get_adj_list(self):
        nodes = self.nodes["node"].tolist()
        adj_list = {node: [] for node in nodes}
        for i in tqdm(range(len(self.edges)), desc="Formating adjacency list"):
            node1 = self.edges["node1"].iloc[i]
            node2 = self.edges["node2"].iloc[i]
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)
        return adj_list

    def _get_tree_info(self, target_idx, tree_height):
        binary_tree_code = format(target_idx, f"0{tree_height}b")
        binary_tree_code = [int(c) for c in binary_tree_code][:-1]
        path_nodes = [0]
        c = 0
        for char in binary_tree_code:
            if char == 0:
                c = c * 2 + 1
            else:
                c = c * 2 + 2
            path_nodes.append(c)
        path_nodes.pop(-1)
        # assert len(path_nodes) == len(binary_tree_code)
        return binary_tree_code, path_nodes

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = [node - 1 for node in sample]
        input_node = sample[self.window_size]
        context_nodes = sample[: self.window_size] + sample[self.window_size + 1 :]
        bin_tree_codes = []  # [s,h]
        path_nodes = []  # [s,h]
        for node in context_nodes:
            b_t_code, p_nodes = self._get_tree_info(node, self.tree_height)
            bin_tree_codes.append(b_t_code)
            path_nodes.append(p_nodes)
        return {
            "input_node": input_node,
            "bin_tree_codes": bin_tree_codes,
            "path_nodes": path_nodes,
        }

    def collate_fn(batch):
        input_nodes = [f["input_node"] for f in batch]
        bin_tree_codes = [f["bin_tree_codes"] for f in batch]
        path_nodes = [f["path_nodes"] for f in batch]
        input_nodes = torch.LongTensor(input_nodes)  # [b]
        bin_tree_codes = torch.tensor(bin_tree_codes)  # [b, s, h]
        path_nodes = torch.tensor(path_nodes)  # [b, s, h]
        return input_nodes, bin_tree_codes, path_nodes


class SkipGramKLDataset(SkipGramHierarchicalSoftmaxDataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        window_size: int,
        walks_num: int,
        t: int,
        bias: int,
    ) -> None:
        super().__init__(nodes_file, edges_file, window_size, walks_num, t, bias)

    def _gaussian_probability_distribution(self, n, mu=0, sigma=1):
        """
        生成一个长度为n的正态分布概率数组，使得中间概率最大，符合正态分布 (使用PyTorch)。
        :param n: 生成的概率分布的长度
        :param mu: 正态分布的均值
        :param sigma: 正态分布的标准差
        :return: 长度为n的概率分布，概率和为1
        """
        # 在 [-3, 3] 区间内生成 n 个点，使用 torch.linspace
        x = torch.linspace(-3, 3, n)

        # 计算正态分布的概率密度值，使用 torch 的张量操作
        pdf = torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
            sigma * torch.sqrt(torch.tensor(2 * torch.pi))
        )

        # 归一化，使得所有概率和为1
        pdf /= torch.sum(pdf)

        return pdf

    def __getitem__(self, index):
        sample = self.data[index]
        sample = [node - 1 for node in sample]
        input_node = sample[self.window_size]
        context_nodes = sample[: self.window_size] + sample[self.window_size + 1 :]
        context_nodes_pdf = self._gaussian_probability_distribution(
            len(context_nodes), mu=0, sigma=1
        )
        prob_dist = torch.zeros([self.nodes.shape[0]]) + 1e-10
        prob_dist[context_nodes] = context_nodes_pdf
        prob_dist = prob_dist / prob_dist.sum()
        return {"input_node": input_node, "prob_dist": prob_dist}

    def collate_fn(batch):
        input_nodes = [f["input_node"] for f in batch]
        prob_dists = [f["prob_dist"] for f in batch]
        input_nodes = torch.LongTensor(input_nodes)
        prob_dists = torch.stack(prob_dists, dim=0)
        return input_nodes, prob_dists


class SkipGramSoftmaxDataset(SkipGramHierarchicalSoftmaxDataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        window_size: int,
        walks_num: int,
        t: int,
        bias: int,
        save_path: str = None,
        load_path: str = None,
        load_path_list: list = None,
    ) -> None:
        super().__init__(
            nodes_file,
            edges_file,
            window_size,
            walks_num,
            t,
            bias,
            save_path,
            load_path,
            load_path_list,
        )

    def __getitem__(self, index):
        sample = self.data[index]
        sample = [node - 1 for node in sample]
        input_node = sample[self.window_size]
        context_nodes = sample[: self.window_size] + sample[self.window_size + 1 :]
        return {"input_node": input_node, "context_node": context_nodes}

    def collate_fn(batch):
        input_nodes = [f["input_node"] for f in batch]
        context_nodes = [f["context_node"] for f in batch]
        input_nodes = torch.LongTensor(input_nodes)
        context_nodes = torch.LongTensor(context_nodes)
        return input_nodes, context_nodes


class LogRecorder:
    def __init__(self, info: str = None, config: dict = None, verbose: bool = False):
        self.info = info
        self.config = config
        self.log = []
        self.verbose = verbose
        self.record = None
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.best_score = None

    def add_log(self, **kwargs):
        if self.verbose:
            print(kwargs)
        self.log.append(kwargs)

    def to_dict(self):
        record = dict()
        record["info"] = self.info
        record["config"] = self.config
        record["log"] = self.log
        record["best_score"] = self.best_score
        record["time"] = self.time
        self.record = record
        return self.record

    def save(self, path):
        if self.record == None:
            self.to_dict()
        with open(path, "w") as f:
            json.dump(self.record, f, ensure_ascii=False)


class SkipGramHierarchicalSoftmaxDataset2(SkipGramHierarchicalSoftmaxDataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        window_size: int,
        walks_num: int,
        t: int,
        bias: int,
        save_path: str = None,
        load_path: str = None,
        load_path_list: list = None,
        path_nodes_path: str = None,
        bin_tree_nodes_path: str = None,
    ) -> None:
        super().__init__(
            nodes_file,
            edges_file,
            window_size,
            walks_num,
            t,
            bias,
            save_path,
            load_path,
            load_path_list,
        )
        self.path_nodes_indices = pickle.load(open(path_nodes_path, "rb"))
        self.bin_tree_nodes = pickle.load(open(bin_tree_nodes_path, "rb"))

    def __getitem__(self, index):
        sample = self.data[index]
        sample = [node - 1 for node in sample]
        input_node = sample[self.window_size]
        context_nodes = sample[: self.window_size] + sample[self.window_size + 1 :]
        bin_tree_codes = []  # [s,h]
        path_nodes = []  # [s,h]
        for node in context_nodes:
            b_t_code = self.bin_tree_nodes[node][
                :-1
            ]  # 最后一个用不到，path_node在预处理数据的时候已经去除了，但是节点编码没有去除
            p_nodes = self.path_nodes_indices[node]
            bin_tree_codes.append(b_t_code)
            path_nodes.append(p_nodes)
        return {
            "input_node": input_node,
            "bin_tree_codes": bin_tree_codes,
            "path_nodes": path_nodes,
        }


def list_all_files_pathlib(dir_path):
    return [str(file) for file in Path(dir_path).rglob("*") if file.is_file()]
