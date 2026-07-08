"""
miniImageNet Episodic Sampler
==============================
实现 Episodic Training Pipeline 的核心：每个 episode 随机采样
N-way K-shot 的 support set 和 query set。

参考: Prototypical Networks for Few-shot Learning (Snell et al., 2017)
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── miniImageNet 标准参数 ───
IMG_SIZE = 84          # 图片统一 resize 到 84×84
MEAN = [0.485, 0.456, 0.406]   # ImageNet 均值
STD  = [0.229, 0.224, 0.225]   # ImageNet 标准差

# 4A 组：miniImageNet 子集划分
N_TRAIN = 64    # 训练类别数
N_VAL   = 16    # 验证类别数
N_TEST  = 20    # 测试类别数


def get_transform(augment: bool = False):
    """获取图像预处理变换

    Args:
        augment: 是否使用数据增强（训练时 True，测试时 False）
    """
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    return transforms.Compose(ops)


class ImageDataset(Dataset):
    """基础数据集：加载指定类别的所有图片"""

    def __init__(self, root: str, class_names: list, augment: bool = False):
        """
        Args:
            root: 数据集根目录（包含类别子文件夹）
            class_names: 要加载的类别名列表
            augment: 是否启用数据增强
        """
        self.transform = get_transform(augment)
        self.samples = []          # [(img_path, class_idx), ...]
        self.class_to_idx = {}     # 类别名 → 编号 (0 ~ N-1)

        for idx, cls in enumerate(sorted(class_names)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            self.class_to_idx[cls] = idx
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

        print(f"  [Dataset] 加载 {len(class_names)} 个类别, {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


class EpisodicSampler:
    """Episodic (N-way K-shot) 采样器 —— 核心任务 4


    """

    def __init__(self, dataset: ImageDataset, n_way: int, n_shot: int, n_query: int):
        """
        Args:
            dataset: ImageDataset 实例
            n_way:  每个 episode 中类别数 (如 5)
            n_shot: 每个类别 support 样本数 (如 1, 5, 10)
            n_query:每个类别 query 样本数 (如 15)
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        # 按类别组织样本索引，加速采样
        self.class_indices = {}
        for cls_name, cls_idx in dataset.class_to_idx.items():
            indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == cls_idx]
            if len(indices) >= (n_shot + n_query):
                self.class_indices[cls_idx] = indices

        self.available_classes = list(self.class_indices.keys())
        if len(self.available_classes) < n_way:
            raise ValueError(f"可用类别数 ({len(self.available_classes)}) 少于 n_way ({n_way})")

    def sample_episode(self):
        """采样一个 episode

        Returns:
            support_images: (n_way * n_shot, C, H, W)
            support_labels: (n_way * n_shot,)
            query_images:   (n_way * n_query, C, H, W)
            query_labels:   (n_way * n_query,)
        """
        # 1. 随机选择 n_way 个类别
        chosen_classes = random.sample(self.available_classes, self.n_way)

        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        # 2. 对每个类别采样 support 和 query
        for new_label, cls_idx in enumerate(chosen_classes):
            pool = self.class_indices[cls_idx]
            sampled = random.sample(pool, self.n_shot + self.n_query)

            support_idx = sampled[:self.n_shot]
            query_idx = sampled[self.n_shot:]

            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_imgs.append(img)
                support_lbls.append(new_label)  # 重新映射到 0~n_way-1

            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_imgs.append(img)
                query_lbls.append(new_label)

        # 3. 堆叠为 batch tensor
        return (
            torch.stack(support_imgs),
            torch.tensor(support_lbls, dtype=torch.long),
            torch.stack(query_imgs),
            torch.tensor(query_lbls, dtype=torch.long),
        )

    def sample_batch(self, batch_size: int):
        """采样一批 episodes

        Returns:
            list of (support_imgs, support_labels, query_imgs, query_labels)
        """
        return [self.sample_episode() for _ in range(batch_size)]


def create_splits(data_root: str):
    """从 ImageNet 子集中划分 miniImageNet 风格的 train/val/test

    全部从 train 目录中取 64+16+20=100 个类别（train 目录每类 ~34张图，足够 episodic 采样）。

    Args:
        data_root: imagenet-mini 的路径

    Returns:
        (train_classes, val_classes, test_classes, train_root, test_root)
    """
    train_root = os.path.join(data_root, 'train')

    # 获取所有可用类别
    all_classes = sorted([d for d in os.listdir(train_root)
                          if os.path.isdir(os.path.join(train_root, d))])

    # 固定随机种子，保证可复现
    rng = random.Random(42)
    rng.shuffle(all_classes)

    train_classes = all_classes[:N_TRAIN]
    val_classes   = all_classes[N_TRAIN:N_TRAIN + N_VAL]
    test_classes  = all_classes[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

    # 所有 split 都从 train_root 读取（图片数量充足）
    print(f"[Split] train: {len(train_classes)} 类 | val: {len(val_classes)} 类 | test: {len(test_classes)} 类")

    return train_classes, val_classes, test_classes, train_root, train_root


if __name__ == '__main__':
    # 快速测试采样逻辑
    data_root = r'C:\Users\28487\Desktop\数据集\imagenet-mini'
    train_classes, val_classes, test_classes, train_root, val_root = create_splits(data_root)

    ds = ImageDataset(train_root, train_classes, augment=True)
    sampler = EpisodicSampler(ds, n_way=5, n_shot=5, n_query=15)

    sup_img, sup_lbl, qry_img, qry_lbl = sampler.sample_episode()
    print(f"\n[Test] Support: {sup_img.shape}, Query: {qry_img.shape}")
    print(f"[Test] Support labels: {sup_lbl}")
    print(f"[Test] Query labels:   {qry_lbl}")
