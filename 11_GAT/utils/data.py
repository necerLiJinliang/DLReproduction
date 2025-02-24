import pandas as pd
import numpy as np
import torch
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_random_seed(42)


class DataPre(object):
    def __init__(self, cites_file_path, content_file_path):
        cites = pd.read_csv(cites_file_path, sep="\t", names=["node1", "node2"])
        content = pd.read_csv(
            content_file_path,
            sep="\t",
            names=["node"] + [f"feat{i}" for i in range(1433)] + ["label"],
        )
        content = content.sort_values("node").reset_index(drop=True)
        nodes = content["node"].unique()
        node2id = {node: i for i, node in enumerate(nodes)}
        self.node2id = node2id
        content["node"] = content["node"].map(node2id)
        cites["node1"] = cites["node1"].map(node2id)
        cites["node2"] = cites["node2"].map(node2id)
        self.nodes = list(range(len(nodes)))
        self.cites = cites
        self.content = content

    def get_features(self):
        features = self.content.iloc[:, 1:-1].values
        return torch.from_numpy(features).float()

    def get_edge_index(self):
        edge_index = [
            self.cites["node1"].tolist() + self.cites["node2"].tolist(),
            self.cites["node2"].tolist() + self.cites["node1"].tolist(),
        ]
        edge_index = torch.LongTensor(edge_index)
        return edge_index

    def get_labels(self, nodes):
        labels_all = self.content["label"].astype("category").cat.codes.values
        labels = labels_all[nodes]
        return torch.from_numpy(labels).long()

    def get_adj_matrix(self):
        adj_matrix = torch.zeros((len(self.node2id), len(self.node2id)))
        adj_matrix[self.cites["node1"], self.cites["node2"]] = 1
        adj_matrix[self.cites["node2"], self.cites["node1"]] = 1
        return adj_matrix

    def get_data(self):
        features = self.get_features()
        labels = self.get_labels()
        adj_matrix = self.get_adj_matrix()
        return {"features": features, "adj_matrix": adj_matrix, "labels": labels}

    def get_supervised_sample(self):
        sampled_nodes = (
            self.content.groupby("label")
            .apply(lambda x: x.sample(n=20, random_state=42))
            .reset_index(drop=True)
        )["node"].tolist()
        labels = self.get_labels(sampled_nodes)
        return sampled_nodes, labels

    def get_test_sample(self, num_test_nodes=2000):
        sample_nodes = random.sample(self.nodes, num_test_nodes)
        labels = self.get_labels(sample_nodes)
        return sample_nodes, labels

    def get_sparse_adj_matrix(self):
        adj_matrix = self.get_adj_matrix()
        return adj_matrix.to_sparse()

    def get_sparse_degree_matrix(self):
        adj_matrix = self.get_adj_matrix()
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        return degree_matrix.to_sparse()
