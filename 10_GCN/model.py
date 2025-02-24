import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, sparse_adj, supervised_nodes, labels):
        a = self.get_normalized_a_matrix(sparse_adj)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(torch.mm(a, x))
        x = self.fc2(x)
        logits = torch.mm(a, x)
        output = (logits,)
        if supervised_nodes is not None:
            supervised_logits = logits[supervised_nodes]
            loss = F.cross_entropy(supervised_logits, labels)
            output = (logits, loss)
        return output

    def get_normalized_a_matrix(
        self,
        sparse_adj_matrix,
    ):
        adj_matrix = sparse_adj_matrix.to_dense()
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
        row_norm = torch.norm(adj_matrix, p=2, dim=-1)
        row_norm_inverse = row_norm.pow(-0.5)
        row_norm_inverse[row_norm == 0] = 0
        col_norm = torch.norm(adj_matrix, p=2, dim=0)
        col_norm_inverse = col_norm.pow(-0.5)
        col_norm_inverse[col_norm == 0] = 0
        a = row_norm_inverse.unsqueeze(1) * adj_matrix * col_norm_inverse.unsqueeze(0)

        return a


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activate, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.activate = activate
        self.dropout = nn.Dropout(dropout)

    def normalize(self, edge_index, num_nodes, edge_weight=None):
        """
        对 edge_index 进行对称归一化
        :param edge_index: [2, num_edges]，边的索引，COO 格式
        :param num_nodes: int，图中节点数
        :param edge_weight: 边的权重（可选），默认全为 1
        :return: (edge_index, edge_weight)，对称归一化后的边索引和权重
        """
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1), device=edge_index.device
            )  # 默认为全 1

        # 计算每个节点的度（Degree）
        row = edge_index[0]
        col = edge_index[1]
        # row, col = edge_index
        deg = torch.zeros(num_nodes, device=edge_index.device)  # 初始化度
        deg.scatter_add_(0, row, edge_weight)  # 累加行节点对应的度

        # 计算 D^{-1/2}，避免度为 0 的情况（防止除 0）
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0  # 对度为 0 的节点保持为 0

        # 对称归一化权重：edge_weight * D^{-1/2}_row * D^{-1/2}_col
        norm_edge_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, norm_edge_weight

    def forward(self, x, edge_index):
        # 线性变换
        x = self.fc(x)
        x = self.dropout(x)
        self_connection = torch.LongTensor([range(x.shape[0]), range(x.shape[0])]).to(
            edge_index
        )
        edge_index = torch.cat([edge_index, self_connection], dim=-1)
        l_n = torch.index_select(x, 0, edge_index[0])  # [m,d]
        # 邻接矩阵汇聚
        norm_edge_weight = self.normalize(edge_index, x.size(0))[1]
        x = scatter_add(
            l_n * norm_edge_weight.unsqueeze(-1),
            index=edge_index[1],
            dim=0,
        )
        if self.activate is not None:
            x = self.activate(x)
        return x


class GCN2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=0.5):
        super(GCN2, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, nn.ReLU(), dropout=dropout)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, None, dropout=dropout)

    def forward(self, x, edge_index, supervised_nodes, labels):
        x = self.gcn1(x, edge_index)
        logits = self.gcn2(x, edge_index)
        output = (logits,)
        if supervised_nodes is not None:
            supervised_logits = logits[supervised_nodes]
            loss = F.cross_entropy(supervised_logits, labels)
            output = (logits, loss)
        return output
