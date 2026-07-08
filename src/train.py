"""
Episodic Training Pipeline（核心任务 3 & 4）
=============================================
实现 Prototypical Network 的 episodic 训练：
每个 episode 模拟一次 few-shot 测试场景来训练模型。

训练技巧: episode采样 + 学习率预热 + 早停
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 确保路径可导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import (ImageDataset, EpisodicSampler, create_splits,
                         get_transform, N_TRAIN, N_VAL, N_TEST)
from src.models.backbone import get_backbone
from src.models.protonet import PrototypicalNetwork, prototypical_loss, accuracy


class WarmupScheduler:
    """学习率预热包装器 —— 前 warmup_steps 步线性增长，之后交给主调度器"""

    def __init__(self, optimizer, warmup_steps: int, base_lr: float, main_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.main_scheduler = main_scheduler
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # 线性预热
            lr = self.base_lr * self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.main_scheduler is not None:
            self.main_scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """早停机制：验证准确率连续 patience 个 epoch 不提升则停止"""

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        """返回 True 表示应早停"""
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop


def train_episode(protonet, sampler, n_way, n_shot, optimizer, device):
    """训练一个 episode

    Returns:
        (loss, accuracy)
    """
    protonet.train()

    # 采样一个 episode
    sup_img, sup_lbl, qry_img, qry_lbl = sampler.sample_episode()
    sup_img, sup_lbl = sup_img.to(device), sup_lbl.to(device)
    qry_img, qry_lbl = qry_img.to(device), qry_lbl.to(device)

    # 前向传播
    logits = protonet(sup_img, sup_lbl, qry_img, n_way, n_shot)
    loss   = prototypical_loss(logits, qry_lbl)
    acc    = accuracy(logits, qry_lbl)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc


@torch.no_grad()
def validate(protonet, sampler, n_way, n_shot, n_episodes, device):
    """验证：在验证集上评估多个 episode 的平均准确率"""
    protonet.eval()
    total_acc = 0.0
    total_loss = 0.0

    for _ in range(n_episodes):
        sup_img, sup_lbl, qry_img, qry_lbl = sampler.sample_episode()
        sup_img, sup_lbl = sup_img.to(device), sup_lbl.to(device)
        qry_img, qry_lbl = qry_img.to(device), qry_lbl.to(device)

        logits = protonet(sup_img, sup_lbl, qry_img, n_way, n_shot)
        loss   = prototypical_loss(logits, qry_lbl)
        acc    = accuracy(logits, qry_lbl)

        total_loss += loss.item()
        total_acc  += acc

    return total_loss / n_episodes, total_acc / n_episodes


def train(config: dict):
    """完整训练流程

    Args:
        config: 训练参数字典（见 main.py）
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[Device] 使用: {device}")

    # ── 1. 数据准备 ──
    data_root = config['data_root']
    train_classes, val_classes, test_classes, train_root, val_root = create_splits(data_root)

    train_ds = ImageDataset(train_root, train_classes, augment=True)
    val_ds   = ImageDataset(train_root, val_classes, augment=False)

    n_way   = config['n_way']
    n_shot  = config['n_shot']
    n_query = config['n_query']

    train_sampler = EpisodicSampler(train_ds, n_way, n_shot, n_query)
    val_sampler   = EpisodicSampler(val_ds, n_way, n_shot, n_query)

    # ── 2. 模型 ──
    backbone = get_backbone(config.get('backbone', 'conv4')).to(device)
    protonet = PrototypicalNetwork(backbone).to(device)

    param_count = sum(p.numel() for p in protonet.parameters() if p.requires_grad)
    print(f"[Model] {config.get('backbone', 'conv4')} — 参数量: {param_count:,}")

    # ── 3. 优化器与调度器 ──
    base_lr = config.get('lr', 1e-3)
    optimizer = Adam(protonet.parameters(), lr=base_lr)
    main_scheduler = CosineAnnealingLR(optimizer,
                                       T_max=config['train_episodes'] - config.get('warmup', 500))
    scheduler = WarmupScheduler(optimizer,
                                warmup_steps=config.get('warmup', 500),
                                base_lr=base_lr,
                                main_scheduler=main_scheduler)
    early_stop = EarlyStopping(patience=config.get('patience', 20))

    # ── 4. 训练循环 ──
    n_train_episodes = config['train_episodes']
    val_episodes     = config.get('val_episodes', 200)
    log_interval     = config.get('log_interval', 100)
    val_interval     = config.get('val_interval', 500)

    train_losses, train_accs = [], []
    val_losses, val_accs     = [], []

    print(f"\n{'='*60}")
    print(f"开始训练: {n_way}-way {n_shot}-shot | {n_train_episodes} episodes")
    print(f"{'='*60}\n")

    start_time = time.time()

    for ep in range(1, n_train_episodes + 1):
        loss, acc = train_episode(protonet, train_sampler, n_way, n_shot, optimizer, device)
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)

        if ep % log_interval == 0:
            avg_loss = np.mean(train_losses[-log_interval:])
            avg_acc  = np.mean(train_accs[-log_interval:])
            lr = scheduler.get_lr()
            print(f"[Ep {ep:5d}/{n_train_episodes}] "
                  f"loss={avg_loss:.4f} | acc={avg_acc:.3f} | lr={lr:.6f}")

        # 验证
        if ep % val_interval == 0 or ep == n_train_episodes:
            val_loss, val_acc = validate(protonet, val_sampler, n_way, n_shot, val_episodes, device)
            val_losses.append((ep, val_loss))
            val_accs.append((ep, val_acc))

            elapsed = time.time() - start_time
            print(f"  [Val  @{ep:5d}] loss={val_loss:.4f} | acc={val_acc:.4f} | "
                  f"best={early_stop.best_acc:.4f} | {elapsed:.0f}s")

            if early_stop.step(val_acc):
                print(f"\n[!] 早停触发于 episode {ep}，最佳验证准确率: {early_stop.best_acc:.4f}")
                break

    # ── 5. 保存模型 ──
    save_dir = config.get('save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)

    tag = config.get('tag', f'{config["backbone"]}_{n_way}way{n_shot}shot')
    model_path = os.path.join(save_dir, f'protonet_{tag}.pth')
    torch.save({
        'model_state_dict': protonet.state_dict(),
        'config': config,
        'val_acc': early_stop.best_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }, model_path)
    print(f"\n[✓] 模型已保存至: {model_path}")

    return model_path, early_stop.best_acc


if __name__ == '__main__':
    config = {
        'data_root': r'C:\Users\28487\Desktop\数据集\imagenet-mini',
        'backbone': 'conv4',
        'n_way': 5,
        'n_shot': 5,
        'n_query': 15,
        'lr': 1e-3,
        'train_episodes': 10000,
        'val_episodes': 200,
        'log_interval': 100,
        'val_interval': 500,
        'warmup': 500,
        'patience': 20,
        'save_dir': './results',
        'tag': 'test',
    }
    train(config)
