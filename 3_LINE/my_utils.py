import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from network import NetworkUnDir


class LINEDataset(Dataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        neg_sample_num: int,
        edge_weight_by_degree: bool,
        sample_num: int,
    ) -> None:
        super().__init__()
        self.neg_sample_num = neg_sample_num
        self.edge_weight_by_degree = edge_weight_by_degree
        self.sample_num = sample_num
        self.network = NetworkUnDir(
            nodes_file=nodes_file,
            edges_file=edges_file,
            neg_sample_num=neg_sample_num,
            edge_weight_by_degree=edge_weight_by_degree,
        )

    def __getitem__(self, index):
        """返回采样结果

        Args:
            index (int): 索引

        Returns:
            dict: {
                "source_node": int,
                "sample_nodes": [int,int,...],
                "sample_labels":[1,0,0,...]
            }
        """
        res = self.network.sample()
        return res

    def __len__(
        self,
    ):
        return self.sample_num

    def collate_fn(batch):
        source_nodes = [f["source_node"] for f in batch]
        sample_nodes = [f["sample_nodes"] for f in batch]
        sample_labels = [f["sample_labels"] for f in batch]
        source_nodes = torch.LongTensor(source_nodes)  # [b]
        sample_nodes = torch.LongTensor(sample_nodes)  # [b, s]
        sample_labels = torch.tensor(sample_labels)  # [b, s]
        return source_nodes, sample_nodes
