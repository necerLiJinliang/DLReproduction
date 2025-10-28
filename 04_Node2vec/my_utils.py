import pandas as pd
from torch.utils.data import Dataset
import torch
import json
from node2vec import Node2Vec
from datetime import datetime
import pickle


class Node2vecDataset(Dataset):
    def __init__(
        self,
        nodes_file,
        edges_file,
        num_walks,
        length_walk,
        num_negtive_samples,
        windom_size,
        p,
        q,
        walks_save_path=None,
        walks_load_path=None,
    ) -> None:
        super().__init__()
        self.node2vec = Node2Vec(
            nodes_file=nodes_file,
            edges_file=edges_file,
            num_walks=num_walks,
            length_walk=length_walk,
            num_negative_samples=num_negtive_samples,
            window_size=windom_size,
            p=p,
            q=q,
        )
        if walks_load_path:
            print(f"Loading walks from {walks_load_path}.")
            self.walks = pickle.load(open(walks_load_path, "rb"))
        else:
            self.walks = self.node2vec.random_walk()
        if walks_save_path:
            print(f"Saving walks in {walks_save_path}")
            with open(walks_save_path, "wb") as f:
                pickle.dump(self.walks, f)
        self.windom_size = windom_size
        self.num_negative_samples = num_negtive_samples

    def __getitem__(self, index):
        walk: list = self.walks[index]
        data_pairs = []
        labels = []
        for i, source_node in enumerate(walk):
            context_node_list = (
                walk[max(0, i - self.windom_size) : i]
                + walk[i + 1 : i + 1 + self.windom_size]
            )
            data_pair = [
                [source_node, context_node] for context_node in context_node_list
            ]
            label = [1] * len(context_node_list)
            neg_nodes = self.node2vec.neg_sampling(
                pos_nodes=[source_node] + context_node_list,
                num_samples=self.num_negative_samples * len(context_node_list),
            )
            data_pair_neg = [[source_node, neg_node] for neg_node in neg_nodes]
            label += [0] * len(data_pair_neg)
            data_pair += data_pair_neg
            data_pairs.extend(data_pair)
            labels.extend(label)
        assert len(data_pairs) == len(labels)
        return data_pairs, labels

    def __len__(
        self,
    ):
        return len(self.walks)

    def collate_fn(batch):
        data_pairs = [f[0] for f in batch]
        labels = [f[1] for f in batch]
        data_pairs = torch.LongTensor(data_pairs)  # [b,n,2]
        source_nodes = data_pairs[:, :, 0]  # [b,n]
        context_nodes = data_pairs[:, :, 1]  # [b,n]
        labels = torch.tensor(labels, dtype=float)  # [b,n]
        return source_nodes, context_nodes, labels


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



    