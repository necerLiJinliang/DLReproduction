import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramHierarchicalSoftmaxModel(nn.Module):
    def __init__(self, nodes_num: int, embedding_dim: int) -> None:
        super().__init__()
        self.nodes_num = nodes_num
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=nodes_num,
            embedding_dim=embedding_dim,
        )
        self.cls = nn.Linear(
            in_features=embedding_dim,
            out_features=nodes_num,
        )

    def forward(self, input_nodes, path_nodes: torch.Tensor, tree_codes: torch.Tensor):

        ############################################
        # 由于没有使用哈夫曼树，所以就没使用MASK了 #
        ############################################
        x = self.embedding(input_nodes)  # [b,e]
        B = x.shape[0]
        logits = self.cls(x)  # [b,n]
        # path_nodes [b, s, h]
        # tree_nodes [b, s, h]
        logits_selected = logits[
            torch.arange(0, B).unsqueeze(-1).unsqueeze(-1), path_nodes
        ]  # [b,h]
        loss = F.binary_cross_entropy_with_logits(
            logits_selected.view(B, -1),
            tree_codes.view(B, -1).float(),
        )
        return loss


class SkipGramKLModel(nn.Module):
    def __init__(self, nodes_num: int, embedding_dim: int) -> None:
        super().__init__()
        self.nodes_num = nodes_num
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=nodes_num,
            embedding_dim=embedding_dim,
        )
        self.cls = nn.Linear(
            in_features=embedding_dim,
            out_features=nodes_num,
        )

    def forward(
        self,
        input_nodes,
        prob_dist,
    ):
        ############################################
        # 由于没有使用哈夫曼树，所以就没使用MASK了 #
        ############################################
        # target_ndoes:[b,h]
        x = self.embedding(input_nodes)  # [b,e]
        B = x.shape[0]
        logits = self.cls(x)  # [b,n]
        prob = torch.softmax(logits, dim=-1)
        loss = F.kl_div(prob_dist.log(), prob, reduction="batchmean")
        # path_nodes [b, s, h]
        # tree_nodes [b, s, h]
        return loss


class SkipGramSoftmaxModel(nn.Module):
    def __init__(self, nodes_num: int, embedding_dim: int) -> None:
        super().__init__()
        self.nodes_num = nodes_num
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=nodes_num,
            embedding_dim=embedding_dim,
        )
        self.cls = nn.Linear(
            in_features=embedding_dim,
            out_features=nodes_num,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_nodes,
        context_nodes,
    ):
        ############################################
        # 由于没有使用哈夫曼树，所以就没使用MASK了 #
        ############################################
        # context_nodes: [b, h]
        x = self.embedding(input_nodes)  # [b,e]
        B, H = context_nodes.shape
        logits = self.cls(x)  # [b,n]
        logits = logits.unsqueeze(dim=1).repeat(1, H, 1)  # [b,n] -> [b,h,n]
        loss = self.criterion(
            logits.reshape(-1, self.nodes_num), context_nodes.reshape(-1)
        )
        return loss
