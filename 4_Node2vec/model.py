import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegativeSamplingModel(nn.Module):
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
        self.criterion = nn.BCEWithLogitsLoss()

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.cls.weight)

    def forward(
        self,
        source_nodes,
        context_nodes,
        labels,
    ):
        # source_nodes [b]
        # context_nodes [b]
        # labels [b,h]
        x = self.embedding(source_nodes)  # [b,e]
        logits = self.cls(x)  # [b,n]
        B = x.shape[0]
        # logits_selected = logits[
        #     torch.arange(0, B).unsqueeze(-1), context_nodes
        # ]
        logits_selected = torch.gather(logits, dim=2, index=context_nodes.unsqueeze(-1))
        loss = self.criterion(logits_selected.view(-1), labels.float().view(-1))
        return loss
