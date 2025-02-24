import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import softmax as pyg_softmax


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        dropout=0.5,
        activate=nn.ELU(),
        head_fusion="cat",
    ):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.trans = nn.Linear(input_dim, output_dim, bias=False)
        self.att_cal = nn.Linear(2 * output_dim // num_heads, 1, bias=True)
        self.activate = activate
        self.head_fusion = head_fusion

    def forward(self, x, edge_index):
        # 添加自连接
        self_connection = torch.LongTensor([range(x.shape[0]), range(x.shape[0])]).to(
            edge_index
        )
        edge_index = torch.cat([edge_index, self_connection], dim=-1)
        # x = self.dropout(x)
        x = self.trans(x).view(
            -1, self.num_heads, self.output_dim // self.num_heads
        )  # [n,num_heads,e // num_heads]
        x = self.dropout(x)
        x_i = torch.index_select(x, 0, edge_index[0])
        x_j = torch.index_select(x, 0, edge_index[1])  # [m,num_heads,e // num_heads]
        x_ij = torch.cat([x_i, x_j], dim=-1)  # [m,num_heads,2 * e // num_heads]
        att_score = self.att_cal(x_ij)  # [m,num_heads,1]
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weight = F.softmax(att_score, dim=0)
        att_score_exp_sum = torch.exp(att_score).sum(
            dim=0, keepdim=True
        )  # [1,num_heads,1]
        node_re_coef = scatter_add(src=torch.exp(att_score), index=edge_index[1], dim=0)
        att_re = (
            att_weight
            * att_score_exp_sum
            / torch.index_select(node_re_coef, 0, edge_index[1])
        )
        # att_re = pyg_softmax(att_score, edge_index[1], num_nodes=x.size(0))
        # att_re = self.dropout(att_re)
        x_i = x_i * att_re  # [m,num_heads,e // num_heads]
        x_i = self.dropout(x_i)
        x = scatter_add(src=x_i, index=edge_index[1], dim=0)
        if self.head_fusion == "cat":
            x = x.view(-1, self.output_dim)
        else:
            x = x.mean(dim=1)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_heads=8,
        dropout=0.5,
    ):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.gat1 = GATLayer(
            input_dim,
            hidden_dim,
            num_heads,
            dropout=dropout,
            activate=nn.ELU(),
            head_fusion="cat",
        )
        self.gat2 = GATLayer(
            hidden_dim,
            output_dim,
            num_heads=1,
            dropout=dropout,
            activate=None,
            head_fusion="mean",
        )
        self.cls = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, supervised_nodes, labels):
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        # logits = self.cls(x)
        logits = x
        output = (logits,)
        if supervised_nodes is not None:
            supervised_logits = logits[supervised_nodes]
            loss = F.cross_entropy(supervised_logits, labels)
            output = (logits, loss)
        return output
