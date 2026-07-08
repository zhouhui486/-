"""
Prototypical Network (原型网络)
================================
实现 Snell et al., 2017 的核心算法。
对每个类，用 support set 嵌入的均值作为"原型"，
query 样本归类到最近的原型。

核心公式:
    p(c) = softmax( -d( f(x), proto_c ) )
    其中 d 是欧氏距离的平方
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """原型网络：包装 backbone，在 forward 中完成原型计算和分类"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def compute_prototypes(self, support_imgs: torch.Tensor, support_labels: torch.Tensor,
                           n_way: int, n_shot: int):
        """计算每个类别的原型向量

        Args:
            support_imgs:   (n_way * n_shot, C, H, W)
            support_labels: (n_way * n_shot,)
            n_way: 类别数
            n_shot:每类样本数

        Returns:
            prototypes: (n_way, D) —— 每个类的原型向量
        """
        # 1. 提取特征: (n_way * n_shot, D)
        z = self.backbone(support_imgs)

        # 2. 按类别聚合取均值
        z = z.reshape(n_way, n_shot, -1)         # (n_way, n_shot, D)
        prototypes = z.mean(dim=1)                 # (n_way, D)
        return prototypes

    def forward(self, support_imgs: torch.Tensor, support_labels: torch.Tensor,
                query_imgs: torch.Tensor, n_way: int, n_shot: int):
        """前向传播：计算 query 的分类 logits

        Args:
            support_imgs:   (n_way * n_shot, C, H, W)
            support_labels: (n_way * n_shot,)
            query_imgs:     (n_way * n_query, C, H, W)
            n_way:  类别数
            n_shot: 每类 support 样本数

        Returns:
            logits:  (n_way * n_query, n_way) —— 每个 query 对每个类的"距离分数"
                     注意: 返回负欧氏距离平方——越大越相似
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_imgs, support_labels, n_way, n_shot)

        # 提取 query 特征
        z_query = self.backbone(query_imgs)          # (n_query * n_way, D)

        # 计算欧氏距离平方: ||z - proto||²
        # (N_q, D) · (D, n_way) = (N_q, n_way)
        dists = torch.cdist(z_query, prototypes, p=2) ** 2

        # 返回负距离作为 logits（之后 softmax 会把更近的映射为更高概率）
        return -dists


def prototypical_loss(logits: torch.Tensor, query_labels: torch.Tensor):
    """计算原型网络损失

    Args:
        logits:       (N_q, n_way) —— protonet.forward 的输出
        query_labels: (N_q,) —— 真实标签 (0 ~ n_way-1)

    Returns:
        loss: 标量 cross-entropy loss
    """
    return F.cross_entropy(logits, query_labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """计算分类准确率"""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


if __name__ == '__main__':
    # 快速测试
    import sys; sys.path.insert(0, '.')
    from src.models.backbone import Conv4

    backbone = Conv4()
    protonet = PrototypicalNetwork(backbone)

    # 模拟 5-way 5-shot, query=15
    n_way, n_shot, n_query = 5, 5, 15
    support = torch.randn(n_way * n_shot, 3, 84, 84)
    query   = torch.randn(n_way * n_query, 3, 84, 84)
    s_lbls  = torch.arange(n_way).repeat_interleave(n_shot)
    q_lbls  = torch.arange(n_way).repeat_interleave(n_query)

    logits = protonet(support, s_lbls, query, n_way, n_shot)
    loss   = prototypical_loss(logits, q_lbls)
    acc    = accuracy(logits, q_lbls)

    print(f"Logits shape: {logits.shape}")      # (75, 5)
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {acc:.4f}")
