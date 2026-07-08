"""
微调基线方法（Fine-tuning Baseline）
=======================================
作为"基于度量的方法 vs 基于微调的方法"对比的对照组。

方案：预训练一个标准分类器 → 测试时在 support set 上微调分类头 →
评估 query set。
"""

import os
import sys
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import (ImageDataset, EpisodicSampler, create_splits,
                         get_transform, N_TRAIN, N_VAL, N_TEST)
from src.models.backbone import get_backbone
from src.models.protonet import accuracy


class Classifier(nn.Module):
    """标准分类器: Backbone + Linear 分类头"""

    def __init__(self, backbone: nn.Module, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(self._get_feat_dim(backbone), n_classes)

    def _get_feat_dim(self, backbone):
        """推断 backbone 输出维度"""
        with torch.no_grad():
            dummy = torch.randn(1, 3, 84, 84)
            return backbone(dummy).shape[1]

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def get_features(self, x):
        """只提取特征，而非分类 logits"""
        return self.backbone(x)


def pretrain_classifier(config: dict):
    """预训练标准分类器（在训练类上用常规分类）

    这一步模拟"有大量标注数据"的场景，
    为后续 few-shot 微调初始权重。
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    data_root = config['data_root']
    train_classes, val_classes, test_classes, train_root, val_root = create_splits(data_root)

    train_ds = ImageDataset(train_root, train_classes, augment=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.get('batch_size', 128), shuffle=True, num_workers=2)

    n_classes = len(train_classes)
    backbone = get_backbone(config.get('backbone', 'conv4')).to(device)
    classifier = Classifier(backbone, n_classes).to(device)

    optimizer = Adam(classifier.parameters(), lr=config.get('lr', 1e-3))
    criterion = nn.CrossEntropyLoss()

    print(f"[Pretrain] 开始在 {n_classes} 个类别上预训练...")

    for epoch in range(config.get('pretrain_epochs', 30)):
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, y)
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_acc  = total_acc / n_batches
        if epoch % 5 == 0 or epoch == config['pretrain_epochs'] - 1:
            print(f"  [Epoch {epoch+1:3d}] loss={avg_loss:.4f} | acc={avg_acc:.4f}")

    # 保存预训练模型
    save_dir = config.get('save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'pretrained_{config.get("backbone", "conv4")}.pth')
    torch.save(classifier.state_dict(), model_path)
    print(f"[✓] 预训练模型已保存至: {model_path}")

    return model_path


def finetune_evaluate(config: dict, test_classes: list, test_root: str):
    """小样本微调评估

    在测试类的 support set 上微调分类头，在 query set 上评估。

    Args:
        config: 配置
        test_classes: 测试类别名列表
        test_root: 测试图片根目录

    Returns:
        (avg_acc, std_acc): 所有测试 episode 的平均准确率和标准差
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    n_way   = config['n_way']
    n_shot  = config['n_shot']
    n_query = config['n_query']

    # 加载预训练模型
    backbone = get_backbone(config.get('backbone', 'conv4')).to(device)
    classifier = Classifier(backbone, N_TRAIN)  # 临时分类头
    pretrained_path = os.path.join(config.get('save_dir', './results'),
                                   f'pretrained_{config.get("backbone", "conv4")}.pth')
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
        # 只加载 backbone 部分
        backbone_state = {k.replace('backbone.', ''): v
                          for k, v in ckpt.items() if k.startswith('backbone.')}
        backbone.load_state_dict(backbone_state, strict=False)
        print(f"[Finetune] 已加载预训练 backbone: {pretrained_path}")
    else:
        print("[Finetune] 警告: 未找到预训练模型，使用随机初始化的 backbone")

    # 创建测试采样器
    test_ds = ImageDataset(test_root, test_classes, augment=False)
    sampler = EpisodicSampler(test_ds, n_way, n_shot, n_query)

    n_episodes = config.get('test_episodes', 600)
    finetune_steps = config.get('finetune_steps', 50)
    finetune_lr = config.get('finetune_lr', 1e-3)

    all_accs = []
    print(f"[Finetune] 测试 {n_episodes} 个 episodes | {n_way}-way {n_shot}-shot | fine-tune {finetune_steps} 步")

    for ep_idx in range(n_episodes):
        sup_img, sup_lbl, qry_img, qry_lbl = sampler.sample_episode()
        sup_img, sup_lbl = sup_img.to(device), sup_lbl.to(device)
        qry_img, qry_lbl = qry_img.to(device), qry_lbl.to(device)

        # 复制模型，在新分类头上微调
        model = copy.deepcopy(backbone)
        feat_dim = model(torch.randn(1, 3, 84, 84).to(device)).shape[1]
        head = nn.Linear(feat_dim, n_way).to(device)

        opt = Adam(list(model.parameters()) + list(head.parameters()), lr=finetune_lr)

        # 在 support set 上微调
        for _ in range(finetune_steps):
            feats = model(sup_img)
            logits = head(feats)
            loss = F.cross_entropy(logits, sup_lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 在 query set 上评估
        with torch.no_grad():
            model.eval()
            feats = model(qry_img)
            logits = head(feats)
            acc = accuracy(logits, qry_lbl)
            all_accs.append(acc)

        if (ep_idx + 1) % 100 == 0:
            print(f"  [Progress] {ep_idx+1}/{n_episodes} — avg acc: {np.mean(all_accs[-100:]):.4f}")

    avg_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    ci95 = 1.96 * std_acc / np.sqrt(len(all_accs))

    print(f"\n[Finetune Result] {n_way}-way {n_shot}-shot: "
          f"{avg_acc:.4f} ± {ci95:.4f} (95% CI)")

    return avg_acc, std_acc


if __name__ == '__main__':
    config = {
        'data_root': r'C:\Users\28487\Desktop\数据集\imagenet-mini',
        'backbone': 'conv4',
        'n_way': 5,
        'n_shot': 5,
        'n_query': 15,
        'lr': 1e-3,
        'batch_size': 128,
        'pretrain_epochs': 30,
        'finetune_steps': 50,
        'finetune_lr': 1e-3,
        'test_episodes': 600,
        'save_dir': './results',
    }

    # 1. 预训练
    pretrain_classifier(config)

    # 2. 微调评估
    _, _, test_classes, _, val_root = create_splits(config['data_root'])
    finetune_evaluate(config, test_classes, val_root)
