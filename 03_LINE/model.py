import torch
import torch.nn as nn
import torch.nn.functional as F


class LINEModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        order,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order
        self.node_embedding = nn.Embedding(
            num_embeddings=num_nodes,
            embedding_dim=embedding_dim,
        )
        if order == 2:
            self.context_embedding = nn.Embedding(
                num_embeddings=num_nodes,
                embedding_dim=embedding_dim,
            )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        if self.order == 2:
            nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, source_nodes, sample_nodes, sample_labels=None):
        source_node_embeddings = self.node_embedding(source_nodes)  # [b, e]
        if self.order == 1:
            sample_node_embeddings = self.node_embedding(sample_nodes)  # [b, s, e]
        else:
            sample_node_embeddings = self.context_embedding(sample_nodes)

        logits = torch.einsum(
            "be,bse->bs",
            source_node_embeddings,
            sample_node_embeddings,
        )  # [b, s]
        pos_loss = -F.logsigmoid(logits[:, 0]).mean()  # [b]
        neg_loss = -F.logsigmoid(-logits[:, 1:]).mean()
        loss = pos_loss + neg_loss
        return loss
