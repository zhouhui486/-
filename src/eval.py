"""
评估与多 shot 对比（核心任务 2 & 5）
======================================
1. 在测试集上评估模型
2. 对比 1-shot / 5-shot / 10-shot 下模型性能变化趋势
3. 对比度量方法 vs 微调方法的性能差异
4. 生成可视化图表
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import (ImageDataset, EpisodicSampler, create_splits,
                         N_TRAIN, N_VAL, N_TEST)
from src.models.backbone import get_backbone
from src.models.protonet import PrototypicalNetwork, accuracy


@torch.no_grad()
def evaluate_protonet(model_path: str, config: dict, shot_list: list = None):
    """评估 Prototypical Network 在不同 shot 数量下的性能

    Args:
        model_path: 训练好的模型路径
        config: 配置字典
        shot_list: 要评估的 shot 列表，如 [1, 5, 10]

    Returns:
        results: {shot: (avg_acc, ci95)} 的字典
    """
    if shot_list is None:
        shot_list = [1, 5, 10]

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # 加载模型
    backbone = get_backbone(config.get('backbone', 'conv4')).to(device)
    protonet = PrototypicalNetwork(backbone).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    protonet.load_state_dict(ckpt['model_state_dict'])
    protonet.eval()
    print(f"[Eval] 已加载模型: {model_path}")
    print(f"[Eval] 训练时验证准确率: {ckpt.get('val_acc', 'N/A')}")

    # 测试数据
    data_root = config['data_root']
    _, _, test_classes, _, val_root = create_splits(data_root)
    test_ds = ImageDataset(val_root, test_classes, augment=False)

    n_way       = config['n_way']
    n_query     = config['n_query']
    n_episodes  = config.get('test_episodes', 600)

    results = {}

    for n_shot in shot_list:
        sampler = EpisodicSampler(test_ds, n_way, n_shot, n_query)
        all_accs = []

        for ep in range(n_episodes):
            sup_img, sup_lbl, qry_img, qry_lbl = sampler.sample_episode()
            sup_img, sup_lbl = sup_img.to(device), sup_lbl.to(device)
            qry_img, qry_lbl = qry_img.to(device), qry_lbl.to(device)

            logits = protonet(sup_img, sup_lbl, qry_img, n_way, n_shot)
            acc = accuracy(logits, qry_lbl)
            all_accs.append(acc)

        avg = np.mean(all_accs)
        ci95 = 1.96 * np.std(all_accs) / np.sqrt(n_episodes)

        results[n_shot] = (avg, ci95)
        print(f"  [ProtoNet] {n_way}-way {n_shot:2d}-shot: {avg:.4f} ± {ci95:.4f}")

    return results


def plot_comparison(protonet_results: dict, finetune_results: dict,
                    n_way: int, save_path: str):
    """绘制对比图表

    Args:
        protonet_results:  {shot: (avg, ci95)}
        finetune_results:  {shot: (avg, ci95)}  可为 None
        n_way: way 数
        save_path: 图片保存路径
    """
    # 设置中文字体（Windows）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    shots = sorted(protonet_results.keys())
    x = np.arange(len(shots))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ─── 子图1: 两类方法绝对对比 ───
    ax = axes[0]
    proto_accs = [protonet_results[s][0] * 100 for s in shots]
    proto_ci95 = [protonet_results[s][1] * 100 for s in shots]

    bars1 = ax.bar(x - width/2, proto_accs, width, label='Prototypical Network (度量)',
                   color='#4C72B0', yerr=proto_ci95, capsize=4)

    if finetune_results:
        ft_accs = [finetune_results.get(s, (0, 0))[0] * 100 for s in shots]
        ft_ci95 = [finetune_results.get(s, (0, 0))[1] * 100 for s in shots]
        ax.bar(x + width/2, ft_accs, width, label='Fine-tuning (微调)',
               color='#DD8452', yerr=ft_ci95, capsize=4)

    ax.set_xlabel('Shot 数量', fontsize=12)
    ax.set_ylabel('分类准确率 (%)', fontsize=12)
    ax.set_title(f'{n_way}-way 分类: 度量 vs 微调', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}-shot' for s in shots])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar, acc in zip(bars1, proto_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ─── 子图2: 性能趋势线 ───
    ax = axes[1]
    ax.plot(shots, proto_accs, 'o-', color='#4C72B0', linewidth=2, markersize=8,
            label='Prototypical Network')
    if finetune_results:
        ax.plot(shots, ft_accs, 's--', color='#DD8452', linewidth=2, markersize=8,
                label='Fine-tuning')

    ax.set_xlabel('Shot 数量', fontsize=12)
    ax.set_ylabel('分类准确率 (%)', fontsize=12)
    ax.set_title(f'{n_way}-way 分类: Shot 数量影响趋势', fontsize=14)
    ax.set_xticks(shots)
    ax.set_xticklabels([f'{s}-shot' for s in shots])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 标注 performance gain
    gain = proto_accs[-1] - proto_accs[0]
    ax.annotate(f'+{gain:.1f}% gain\n(1-shot → {shots[-1]}-shot)',
                xy=(shots[-1], proto_accs[-1]),
                xytext=(shots[-1] - 0.5, proto_accs[-1] - 5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='#333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] 图表已保存至: {save_path}")
    plt.close()


def generate_report(protonet_results: dict, finetune_results: dict,
                    config: dict, save_dir: str):
    """生成实验报告（JSON）"""
    report = {
        'method': 'Prototypical Network vs Fine-tuning Baseline',
        'dataset': 'miniImageNet 子集 (4A组)',
        'backbone': config.get('backbone', 'conv4'),
        'n_way': config['n_way'],
        'results': {
            'prototypical_network': {f'{s}-shot': {'acc': float(r[0]), 'ci95': float(r[1])}
                                     for s, r in protonet_results.items()},
        },
    }
    if finetune_results:
        report['results']['finetune_baseline'] = {
            f'{s}-shot': {'acc': float(r[0]), 'ci95': float(r[1])}
            for s, r in finetune_results.items()
        }

    # 计算对比分析
    if finetune_results:
        report['comparison'] = {}
        for s in protonet_results:
            diff = protonet_results[s][0] - finetune_results[s][0]
            report['comparison'][f'{s}-shot'] = {
                'difference': float(diff),
                'winner': 'protonet' if diff > 0 else 'finetune'
            }

    report_path = os.path.join(save_dir, 'experiment_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[✓] 报告已保存至: {report_path}")

    return report


if __name__ == '__main__':
    # 单独测试评估功能
    config = {
        'data_root': r'C:\Users\28487\Desktop\数据集\imagenet-mini',
        'backbone': 'conv4',
        'n_way': 5,
        'n_shot': 5,
        'n_query': 15,
        'test_episodes': 600,
        'save_dir': './results',
    }

    results = evaluate_protonet('./results/protonet_test.pth', config,
                                shot_list=[1, 5, 10])
    plot_comparison(results, None, 5, './results/comparison.png')
