import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2VectorModelHierarchicalSoftmax(nn.Module):
    def __init__(
        self, words_num: int, embedding_dim: int, no_leaf_nodes_num: int
    ) -> None:
        super().__init__()
        self.words_num = words_num
        self.embedding_dim = embedding_dim
        self.no_leaf_nodes_num = no_leaf_nodes_num

        self.embedding = nn.Linear(
            in_features=words_num,
            out_features=embedding_dim,
            bias=False,
        )
        # self.cls = nn.Parameter(torch.randn([no_leaf_nodes_num, embedding_dim]))
        # with torch.no_grad():
        #     self.cls[-1].zero_()
        self.cls = nn.Linear(embedding_dim, words_num, bias=False)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs_vector, path_nodes_indices, huffman_codes, mask):
        # inputs_vector [b, n]
        # labels [[node index,],...]
        x = self.embedding(inputs_vector)
        batch_size = inputs_vector.shape[0]
        # for i in range(batch_size):
        #     path_node_index = torch.LongTensor(path_nodes_indices[i]).to(x)
        #     path_vector = self.cls[path_nodes_indices[i]]
        #     logits = x[i] @ path_vector.T
        #     # prob = torch.sigmoid(logits)
        #     """
        #         $$
        #         \mathcal{L}(w,j)=(1-d_{j}^{w})\cdot\log[\sigma(\mathbf{x}_{w}^{\top}\theta_{j-1}^{w})]+d_{j}^{w}\cdot\log[1-\sigma(\mathbf{x}_{w}^{\top}\theta_{j-1}^{w})]
        #         $$
        #     """
        #     huffman_code = torch.tensor(huffman_codes[i]).to(x)
        #     loss = loss + F.binary_cross_entropy_with_logits(logits, huffman_code).to(x)
        # loss = loss / batch_size
        
        
        path_nodes_indices = torch.tensor(path_nodes_indices, dtype=torch.long)  # [b,s]
        # path_vector = self.cls[path_nodes_indices]  # [b,s,e]
        # logits = torch.einsum("be,bse->bs", x, path_vector)  # [b,s]
        huffman_codes = torch.tensor(huffman_codes).to(x)
        logits = self.cls(x) # [b, n]
        logits = logits[torch.arange(0,batch_size).unsqueeze(-1),path_nodes_indices]
        # prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, huffman_codes, reduce=False)
        masked_loss = loss * mask
        loss = masked_loss.sum(dim=-1) / mask.sum(dim=-1)
        loss = loss.mean()
        return loss


class Word2VectorModelSoftmax(nn.Module):
    def __init__(
        self, words_num: int, embedding_dim: int, no_leaf_nodes_num: int
    ) -> None:
        super().__init__()
        self.words_num = words_num
        self.embedding_dim = embedding_dim
        self.no_leaf_nodes_num = no_leaf_nodes_num

        self.embedding = nn.Linear(
            in_features=words_num,
            out_features=embedding_dim,
            bias=False,
        )
        self.cls = nn.Linear(embedding_dim, words_num, bias=False)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs_vector, target_ids):
        # inputs_vector [b, n]
        # labels [[node index,],...]
        x = self.embedding(inputs_vector)
        output = self.cls(x)
        loss = self.criterion(output, target_ids)
        return loss


class Word2VectorNegSampling(nn.Module):
    def __init__(
        self,
        words_num: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.words_num = words_num
        self.embedding_dim = embedding_dim

        self.embedding = nn.Linear(
            in_features=words_num,
            out_features=embedding_dim,
            bias=False,
        )
        self.cls = nn.Linear(embedding_dim, words_num, bias=False)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, inputs_vector, chosen_ids, labels):
        # inputs_vector [b, n]
        # labels [[node index,],...]
        x = self.embedding(inputs_vector)
        output = self.cls(x)
        chosen_output = output[torch.arange(0, x.shape[0]).unsqueeze(-1), chosen_ids]
        loss = self.criterion(chosen_output, labels.float())
        return loss
