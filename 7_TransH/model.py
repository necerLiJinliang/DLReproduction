import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


# 定义 TransH 模型
class TransH(nn.Module):
    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim,
        margin=1.0,
        norm=1,
        c=1,
        epsilon=0.0000001,
    ):
        super(TransH, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.norm = norm
        self.norm_vector = nn.Embedding(num_relations, embedding_dim=embedding_dim)
        self.c = c
        self.epsilon = epsilon
        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)
        self.criterion = nn.MarginRankingLoss(margin=margin)

    def project(self, h_emb, t_emb, norm_vector):
        h_emb_proj = (
            torch.sum(h_emb * norm_vector, dim=-1, keepdim=True) * norm_vector
        )  # [b,c,e]
        t_emb_proj = torch.sum(t_emb * norm_vector, dim=-1, keepdim=True) * norm_vector
        return h_emb_proj, t_emb_proj

    def distance(self, h, r, t, norm_vector):
        h_proj, t_proj = self.project(h, t, norm_vector)
        return torch.norm(h - h_proj + r - (t - t_proj), p=self.norm, dim=-1)  # b

    def distance_neg(self, h, r, t, norm_vector):
        # h [b,c,e]
        # t [b,c,e]
        B, C, E = h.shape
        # h = h.reshape(B * C, -1)
        # t = t.reshape(B * C, -1)
        h_proj, t_proj = self.project(h, t, norm_vector.unsqueeze(dim=1))
        return torch.norm(
            h - h_proj + r.unsqueeze(dim=1) - (t - t_proj), p=self.norm, dim=-1
        )

    def forward(self, h, r, t, neg_samples):
        # 正样本
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        neg_h_ids = neg_samples[:, :, 0]  # [b,c]
        neg_t_ids = neg_samples[:, :, 1]  # [b,c]

        # 负样本，和TranE不一样，TransH采取了概率采样的方式，所以每次就弄一个负样本
        neg_h_emb = self.entity_embeddings(neg_h_ids)
        neg_t_emb = self.entity_embeddings(neg_t_ids)
        norm_vector = F.normalize(self.norm_vector(r), p=self.norm, dim=1)

        # 距离计算
        pos_distance = self.distance(h_emb, r_emb, t_emb, norm_vector)  # [b]
        neg_distance = self.distance_neg(
            neg_h_emb, r_emb, neg_t_emb, norm_vector
        )  # [b,c]
        loss_scale = self.scale_loss(
            h_emb,
            t_emb,
            neg_h_emb.reshape(-1, neg_h_emb.shape[-1]),
            neg_t_emb.reshape(-1, neg_t_emb.shape[-1]),
        )
        loss_orthogonal = self.orthogonal_loss(norm_vector, r_emb)
        loss_scale_rel = torch.mean(torch.relu((r_emb**2).sum(dim=-1) - 1))
        loss = self.loss(pos_distance.unsqueeze(dim=-1), neg_distance) + self.c * (
            loss_orthogonal + loss_scale + loss_scale_rel
        )
        return loss

    def scale_loss(self, h_emb, t_emb, neg_h_emb, neg_t_emb):
        embs = torch.cat([h_emb, t_emb, neg_h_emb, neg_t_emb], dim=0)
        loss = torch.mean(torch.relu((embs**2).sum(dim=1) - 1))
        return loss

    def orthogonal_loss(self, norm_vector, rel_emb):
        # loss = torch.sum(norm_vector * rel_emb, dim=1) / (
        #     torch.norm(rel_emb, dim=-1, p=2) ** 2
        # )
        # loss = torch.mean(torch.relu(loss - self.epsilon**2))
        loss = torch.mean((torch.sum(norm_vector * rel_emb, dim=1)) ** 2)
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
