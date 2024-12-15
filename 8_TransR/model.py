import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


# 定义 TransH 模型
class TransR(nn.Module):
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
        super(TransR, self).__init__()
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
        nn.init.xavier_uniform_(self.relation_proj.weight.data)
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self._projs_init()

    def _projs_init(self):
        identity = torch.zeros(self.entity_dim, self.relation_dim)
        for i in range(min(self.entity_dim, self.relation_dim)):
            identity[i][i] = 1
        identity = identity.view(self.entity_dim * self.relation_dim)
        for i in range(self.num_relations):
            self.relation_proj.weight.data[i] = identity

    def project(self, h_emb: torch.Tensor, t_emb: torch.Tensor, rel_proj: torch.Tensor):
        """根据关系投影矩阵对向量进行投影
        Args:
            h_emb (torch.Tensor): [b,k] $$\mathbf{H}$$ 头实体嵌入
            t_emb (torch.Tensor): [b,k] $$\mathbf{T}$$尾实体嵌入
            rel_proj (_type_): [b,k*d] $$\mathbf{M}_r$$  关系投影矩阵向量形式，需要转换成矩阵形式 \n
            $$ \mathbf{h}_r = \mathbf{h}\mathbf{M}_r,\quad \mathbf{t}_r = \mathbf{t}\mathbf{M}_r$$
        Returns:
            tuple: h_emb_proj, t_embe_proj
        """
        ## 为了对负样本[b,m,k]形状的张量也适用，首先对张量维度进行扩展
        if len(h_emb.shape) < 2:
            h_emb = h_emb.unsqueeze(dim=1)
            t_emb = h_emb.unsqueeze(dim=1)
        rel_proj = rel_proj.reshape(-1, self.entity_dim, self.relation_dim)  # [b,k,d]
        h_emb_proj = torch.einsum("bmk,bkd->bmd", h_emb, rel_proj)
        t_emb_proj = torch.einsum("bmk,bkd->bmd", t_emb, rel_proj)
        return h_emb_proj, t_emb_proj

    def distance(self, h_proj: torch.Tensor, r: torch.Tensor, t_proj: torch.Tensor):
        """根据实体投影和关系偏移向量计算距离分数

        Args:
            h_proj (torch.Tensor): [b,m,d] 头实体投影向量
            r (torch.Tensor): [b,m,d] 关系平移向量
            t_proj (torch.Tensor): [b,m,d] 尾实体投影向量

        Returns:
            torch.Tensor: [b,m] 距离分数
        """
        return torch.norm(h_proj + r - t_proj, p=self.norm, dim=-1)  # [b,m]

    def forward(self, h, r, t, neg_samples):
        # 正样本实体嵌入
        h_emb = self.entity_embeddings(h).unsqueeze(dim=1)  # [b,1,k]
        t_emb = self.entity_embeddings(t).unsqueeze(dim=1)  # [b,1,k]
        # 关系平移向量和关系投影矩阵
        r_emb = self.relation_embeddings(r).unsqueeze(dim=1)  # [b,1,k]
        rel_proj = self.relation_proj(r)  # [b,k*d]
        # 负样本实体嵌入
        neg_h_ids = neg_samples[:, :, 0]  # [b,m]
        neg_t_ids = neg_samples[:, :, 1]  # [b,m]
        neg_h_emb = self.entity_embeddings(neg_h_ids)  # [b,m,k]
        neg_t_emb = self.entity_embeddings(neg_t_ids)  # [b,m,k]
        # 正样本投影向量
        h_emb_proj, t_emb_proj = self.project(
            h_emb=h_emb, t_emb=t_emb, rel_proj=rel_proj
        )  # [b,1,d] [b,1,d]
        # 负样本投影向量
        neg_h_emb_proj, neg_t_emb_proj = self.project(
            h_emb=neg_h_emb, t_emb=neg_t_emb, rel_proj=rel_proj
        )
        # 距离计算
        pos_distance = self.distance(h_emb_proj, r_emb, t_emb_proj)  # [b]
        neg_distance = self.distance(neg_h_emb_proj, r_emb, neg_t_emb_proj)  # [b,c]
        loss_scale = self.scale_loss(
            h_emb.reshape(-1, h_emb.shape[-1]),
            t_emb.reshape(-1, h_emb.shape[-1]),
            neg_h_emb.reshape(-1, neg_h_emb.shape[-1]),
            neg_t_emb.reshape(-1, neg_t_emb.shape[-1]),
        )
        loss_projection_scale = self.scale_loss(
            h_emb_proj.reshape(-1, h_emb_proj.shape[-1]),
            t_emb_proj.reshape(-1, t_emb_proj.shape[-1]),
            neg_h_emb_proj.reshape(-1, neg_h_emb_proj.shape[-1]),
            neg_t_emb_proj.reshape(-1, neg_t_emb_proj.shape[-1]),
        )
        rel_scale_loss = torch.mean(
            torch.relu((r_emb.reshape(-1, r_emb.shape[-1]) ** 2).sum(dim=-1) - 1)
        )
        loss = self.loss(pos_distance.unsqueeze(dim=-1), neg_distance) + self.c * (
            loss_scale + loss_projection_scale + rel_scale_loss
        )
        return loss

    def scale_loss(self, h_emb, t_emb, neg_h_emb, neg_t_emb):
        embs = torch.cat([h_emb, t_emb, neg_h_emb, neg_t_emb], dim=0)
        loss = torch.mean(torch.relu((embs**2).sum(dim=-1) - 1))
        return loss

    def loss(self, pos_distance, neg_distance):
        margin_loss = torch.mean(torch.relu(pos_distance + self.margin - neg_distance))
        loss = margin_loss
        # loss = margin_loss
        return loss


class CTransR(TransR):
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
        alpha=0.01,
        num_cluster=4,
        cluster_ids_matrix=None,
    ):
        super(CTransR, self).__init__(
            num_entities,
            num_relations,
            entity_dim,
            relation_dim,
            margin,
            norm,
            c,
            epsilon,
        )
        self.alpha = alpha
        self.num_cluster = num_cluster
        self.cluster_relation_embeddings = nn.Parameter(
            torch.randn([num_relations, num_cluster, relation_dim])
        )
        nn.init.xavier_uniform_(self.cluster_relation_embeddings)
        self.cluster_ids_matrix = cluster_ids_matrix

    def init_cluster_relation_embeddings(self):
        with torch.no_grad():
            self.cluster_relation_embeddings[:, :, :] = (
                self.relation_embeddings.weight.data.unsqueeze(dim=1).repeat(
                    1, self.num_cluster, 1
                )
            )

    def constrain_distance(self, r_emb, r_emb_cluster):
        """计算聚类簇的关系平移向量和全局关系平移向量的之间的约束距离，避免之间相差过大

        Args:
            r_emb (tensor): [b,m,k] 全局关系平移向量
            r_emb_cluster (tensor): [b,m,k] 聚类簇关系平移向量
        Returns:
            tensor: [b] distance
        """
        distance = torch.norm(
            r_emb - r_emb_cluster,
            dim=-1,
            p=self.norm,
        ).squeeze()
        return distance

    def rel_scale_loss(self, r_emb):
        loss = torch.mean(
            torch.relu((r_emb.reshape(-1, r_emb.shape[-1]) ** 2).sum(dim=-1) - 1)
        )
        return loss

    def forward(self, h, r, t, neg_samples):
        """前向传播

        Args:
            h (tensor): [b,]
            r (tensor): [b,]
            t (tenosr): [b,]
            neg_samples (tensor): [b, m, 2]
            首先根据实体的索引判断聚类的结果，获取cluster_ids，然后到cluster_relation_embeddings层中去取对应的关系平移向量。
            最后再计算距离等。
        Returns:
            tensor: loss
        """
        # 正样本实体嵌入
        h_emb = self.entity_embeddings(h).unsqueeze(dim=1)  # [b,1,k]
        t_emb = self.entity_embeddings(t).unsqueeze(dim=1)  # [b,1,k]
        # 负样本实体嵌入
        neg_h_ids = neg_samples[:, :, 0]  # [b,m]
        neg_t_ids = neg_samples[:, :, 1]  # [b,m]
        neg_h_emb = self.entity_embeddings(neg_h_ids)  # [b,m,k]
        neg_t_emb = self.entity_embeddings(neg_t_ids)  # [b,m,k]

        # 获取正样本对应的关系平移向量
        ## 首先获取每个实体对的聚类簇
        cluster_ids_pos = self.cluster_ids_matrix[h, t]
        ## 根据关系id，聚类簇id获取对应的关系平移向量
        r_emb_cluster_pos = self.cluster_relation_embeddings[
            r, cluster_ids_pos
        ].unsqueeze(
            dim=1
        )  # [b,1,k]

        # 获取负样本对应的关系平移向量
        B, M = neg_h_ids.shape
        neg_h_ids_flatten = neg_h_ids.reshape(-1)  # [n]
        neg_t_ids_flatten = neg_t_ids.reshape(-1)  # [n]
        r_neg = r.unsqueeze(dim=1).repeat(1, M).reshape(-1)
        cluster_ids_neg = self.cluster_ids_matrix[
            neg_h_ids_flatten, neg_t_ids_flatten
        ]  # [n]
        r_emb_cluster_neg = self.cluster_relation_embeddings[
            r_neg, cluster_ids_neg
        ].reshape(
            B, M, -1
        )  # [b,m,k]
        # 获取全局的关系平移向量
        r_emb = self.relation_embeddings(r).unsqueeze(dim=1)  # [b,1,k]

        # 获取关系投影矩阵
        rel_proj = self.relation_proj(r)  # [b,k*d]
        # 正样本投影向量
        h_emb_proj, t_emb_proj = self.project(
            h_emb=h_emb, t_emb=t_emb, rel_proj=rel_proj
        )  # [b,1,d] [b,1,d]
        # 负样本投影向量
        neg_h_emb_proj, neg_t_emb_proj = self.project(
            h_emb=neg_h_emb, t_emb=neg_t_emb, rel_proj=rel_proj
        )
        pos_distance = self.distance(h_emb_proj, r_emb_cluster_pos, t_emb_proj)  # [b]
        neg_distance = self.distance(
            neg_h_emb_proj, r_emb_cluster_neg, neg_t_emb_proj
        )  # [b,c]
        pos_constrain_distance = self.constrain_distance(
            r_emb=r_emb, r_emb_cluster=r_emb_cluster_pos
        )
        neg_constrain_distance = self.constrain_distance(
            r_emb=r_emb, r_emb_cluster=r_emb_cluster_neg
        )
        pos_distance = pos_distance + self.alpha * pos_constrain_distance
        neg_distance = neg_distance + self.alpha * neg_constrain_distance
        loss_scale = self.scale_loss(
            h_emb.reshape(-1, h_emb.shape[-1]),
            t_emb.reshape(-1, h_emb.shape[-1]),
            neg_h_emb.reshape(-1, neg_h_emb.shape[-1]),
            neg_t_emb.reshape(-1, neg_t_emb.shape[-1]),
        )
        loss_projection_scale = self.scale_loss(
            h_emb_proj.reshape(-1, h_emb_proj.shape[-1]),
            t_emb_proj.reshape(-1, t_emb_proj.shape[-1]),
            neg_h_emb_proj.reshape(-1, neg_h_emb_proj.shape[-1]),
            neg_t_emb_proj.reshape(-1, neg_t_emb_proj.shape[-1]),
        )
        rel_scale_loss = (
            self.rel_scale_loss(r_emb)
            + self.rel_scale_loss(r_emb_cluster_neg)
            + self.rel_scale_loss(r_emb_cluster_pos)
        )
        loss = (
            self.loss(pos_distance.unsqueeze(dim=-1), neg_distance)
            + self.c * (loss_scale + loss_projection_scale + rel_scale_loss)
            # + 0.001 * (pos_constrain_distance.mean() + neg_constrain_distance.mean())
        )
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
