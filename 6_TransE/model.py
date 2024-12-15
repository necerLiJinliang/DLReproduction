import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class TransE(nn.Module):
    def __init__(
        self,
        num_entities,
        num_relations,
        entity_dim,
        relation_dim,
        margin=1.0,
        norm=1,
        c=1,
        epsilon=0.0000001,
    ):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, entity_dim)
        self.margin = margin
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.norm = norm
        self.relation_proj = nn.Embedding(
            num_relations, embedding_dim=entity_dim * relation_dim
        )
        self.c = c
        self.epsilon = epsilon
        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        self.criterion = nn.MarginRankingLoss(margin=margin)

    def distance(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        """根据实体投影和关系偏移向量计算距离分数

        Args:
            h (torch.Tensor): [b,m,d] 头实体投影向量
            r (torch.Tensor): [b,m,d] 关系平移向量
            t (torch.Tensor): [b,m,d] 尾实体投影向量

        Returns:
            torch.Tensor: [b,m] 距离分数
        """
        return torch.norm(h + r - t, p=self.norm, dim=-1)  # [b,m]

    def forward(self, h, r, t, neg_samples):
        # 正样本实体嵌入
        h_emb = self.entity_embeddings(h).unsqueeze(dim=1)  # [b,1,k]
        t_emb = self.entity_embeddings(t).unsqueeze(dim=1)  # [b,1,k]
        # 关系平移向量和关系投影矩阵
        r_emb = self.relation_embeddings(r).unsqueeze(dim=1)  # [b,1,k]
        # 负样本实体嵌入
        neg_h_ids = neg_samples[:, :, 0]  # [b,m]
        neg_t_ids = neg_samples[:, :, 1]  # [b,m]
        neg_h_emb = self.entity_embeddings(neg_h_ids)  # [b,m,k]
        neg_t_emb = self.entity_embeddings(neg_t_ids)  # [b,m,k]
        # 距离计算
        pos_distance = self.distance(h_emb, r_emb, t_emb)  # [b]
        neg_distance = self.distance(neg_h_emb, r_emb, neg_t_emb)  # [b,c]
        loss_scale = self.scale_loss(
            h_emb.reshape(-1, h_emb.shape[-1]),
            t_emb.reshape(-1, h_emb.shape[-1]),
            neg_h_emb.reshape(-1, neg_h_emb.shape[-1]),
            neg_t_emb.reshape(-1, neg_t_emb.shape[-1]),
        )
        loss = self.loss(pos_distance.unsqueeze(dim=-1), neg_distance) + self.c * (
            loss_scale
        )
        return loss

    def scale_loss(self, h_emb, t_emb, neg_h_emb, neg_t_emb):
        embs = torch.cat([h_emb, t_emb, neg_h_emb, neg_t_emb], dim=0)
        loss = torch.mean(torch.relu((embs**2).sum(dim=-1) - 1))
        return loss

    def loss(self, pos_distance, neg_distance):
        margin_loss = torch.mean(torch.relu(pos_distance + self.margin - neg_distance))
        # margin_loss = self.criterion(pos_distance, neg_distance, labels)
        # loss = margin_loss + self.c * (0.000001 * norm_loss + orthogonal_loss)
        loss = margin_loss
        # loss = margin_loss
        return loss


if __name__ == "__main__":

    # 数据加载器
    def generate_negative_samples(num_entities, batch_size):
        return torch.randint(0, num_entities, (batch_size,), dtype=torch.long)

    # 配置参数
    num_entities = 1000  # 实体数量
    num_relations = 100  # 关系数量
    embedding_dim = 50  # 嵌入维度
    margin = 1.0
    learning_rate = 0.01
    num_epochs = 100

    # 模型和优化器
    model = TransE(num_entities, num_relations, embedding_dim, margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 模拟数据
    batch_size = 64
    head = torch.randint(0, num_entities, (batch_size,))
    relation = torch.randint(0, num_relations, (batch_size,))
    tail = torch.randint(0, num_entities, (batch_size,))

    # 训练
    for epoch in range(num_epochs):
        negative_head = generate_negative_samples(num_entities, batch_size)
        negative_tail = generate_negative_samples(num_entities, batch_size)

        pos_distance, neg_distance_h, neg_distance_t = model(
            head, relation, tail, negative_head, negative_tail
        )
        loss = model.loss(pos_distance, neg_distance_h, neg_distance_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
