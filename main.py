"""
小样本图像分类系统 — 主入口
=============================
4A组: miniImageNet 子集（通用物体）

用法:
    # 1. 训练 Prototypical Network (5-way 5-shot)
    python main.py --mode train --shot 5

    # 2. 训练所有 shot 配置 (1, 5, 10)
    python main.py --mode train_all

    # 3. 评估并生成对比报告
    python main.py --mode eval

    # 4. 完整流程: 预训练 + 训练 ProtoNet + 微调 + 评估
    python main.py --mode full
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train import train
from src.finetune_baseline import pretrain_classifier, finetune_evaluate
from src.eval import evaluate_protonet, plot_comparison, generate_report
from src.dataset import create_splits


def build_config(args, n_shot: int = 5) -> dict:
    """根据命令行参数构建配置"""
    return {
        'data_root': args.data_root,
        'backbone': args.backbone,
        'n_way': args.n_way,
        'n_shot': n_shot,
        'n_query': args.n_query,
        'lr': args.lr,
        'train_episodes': args.train_episodes,
        'val_episodes': args.val_episodes,
        'log_interval': args.log_interval,
        'val_interval': args.val_interval,
        'warmup': args.warmup,
        'patience': args.patience,
        'pretrain_epochs': args.pretrain_epochs,
        'finetune_steps': args.finetune_steps,
        'finetune_lr': args.finetune_lr,
        'test_episodes': args.test_episodes,
        'batch_size': args.batch_size,
        'save_dir': args.save_dir,
        'tag': f'{args.backbone}_{args.n_way}way{n_shot}shot',
    }


def main():
    parser = argparse.ArgumentParser(description='小样本图像分类系统 — 4A组')

    # 运行模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_all', 'eval', 'full'],
                        help='运行模式')

    # 数据参数
    parser.add_argument('--data_root', type=str,
                        default=r'C:\Users\28487\Desktop\数据集\imagenet-mini',
                        help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='conv4',
                        choices=['conv4', 'resnet12'],
                        help='CNN 骨干网络')
    parser.add_argument('--n_way', type=int, default=5,
                        help='每个 episode 的类别数')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='每类 support 样本数')
    parser.add_argument('--n_query', type=int, default=15,
                        help='每类 query 样本数')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--train_episodes', type=int, default=10000,
                        help='训练 episode 总数')
    parser.add_argument('--val_episodes', type=int, default=200,
                        help='每次验证的 episode 数')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='打印日志的 episode 间隔')
    parser.add_argument('--val_interval', type=int, default=500,
                        help='验证的 episode 间隔')
    parser.add_argument('--warmup', type=int, default=500,
                        help='学习率预热步数')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')

    # 预训练 & 微调参数
    parser.add_argument('--pretrain_epochs', type=int, default=30,
                        help='标准分类预训练的 epoch 数')
    parser.add_argument('--finetune_steps', type=int, default=50,
                        help='微调步数')
    parser.add_argument('--finetune_lr', type=float, default=1e-3,
                        help='微调学习率')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='预训练的 batch size')

    # 评估参数
    parser.add_argument('--test_episodes', type=int, default=600,
                        help='测试 episode 数')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    shot_list = [1, 5, 10]  # 核心任务5: 多 shot 对比

    print(f"""
╔══════════════════════════════════════════════════╗
║     小样本图像分类系统 — Few-Shot Learning      ║
║     4A组: miniImageNet 子集（通用物体）          ║
║     Prototypical Network + Fine-tuning Baseline  ║
╚══════════════════════════════════════════════════╝

  数据集: {args.data_root}
  Backbone: {args.backbone}
  模式: {args.mode}
""")

    # ============================================================
    # 模式1: 单次训练
    # ============================================================
    if args.mode == 'train':
        config = build_config(args, args.n_shot)
        model_path, best_acc = train(config)

    # ============================================================
    # 模式2: 训练所有 shot 配置
    # ============================================================
    elif args.mode == 'train_all':
        model_paths = {}
        for shot in shot_list:
            print(f"\n{'#'*60}")
            print(f"#  训练 {args.n_way}-way {shot}-shot")
            print(f"{'#'*60}")
            config = build_config(args, shot)
            config['tag'] = f'{args.backbone}_{args.n_way}way{shot}shot'
            model_path, acc = train(config)
            model_paths[shot] = model_path
        print(f"\n[✓] 所有模型训练完成:")
        for s, p in model_paths.items():
            print(f"    {s}-shot: {p}")

    # ============================================================
    # 模式3: 评估
    # ============================================================
    elif args.mode == 'eval':
        config = build_config(args, args.n_shot)

        protonet_results = {}
        finetune_results = {}

        # 评估 ProtoNet（每个 shot 一个模型）
        for shot in shot_list:
            model_path = os.path.join(
                args.save_dir, f'protonet_{args.backbone}_{args.n_way}way{shot}shot.pth')

            if os.path.exists(model_path):
                shot_config = build_config(args, shot)
                res = evaluate_protonet(model_path, shot_config, shot_list=[shot])
                protonet_results[shot] = res[shot]
            else:
                print(f"[!] 模型不存在，跳过: {model_path}")

        # 评估微调基线
        _, _, test_classes, _, val_root = create_splits(args.data_root)
        for shot in shot_list:
            shot_config = build_config(args, shot)
            acc, std = finetune_evaluate(shot_config, test_classes, val_root)
            ci95 = 1.96 * std / np.sqrt(shot_config['test_episodes'])
            finetune_results[shot] = (acc, ci95)

        # 绘图 & 报告
        if protonet_results:
            save_path = os.path.join(args.save_dir, 'comparison.png')
            plot_comparison(protonet_results, finetune_results, args.n_way, save_path)
            generate_report(protonet_results, finetune_results, config, args.save_dir)

    # ============================================================
    # 模式4: 完整流程
    # ============================================================
    elif args.mode == 'full':
        config = build_config(args, args.n_shot)
        print("Phase 1/3: 预训练标准分类器\n")

        pretrain_classifier(config)

        print("\n" + "="*60)
        print("Phase 2/3: 训练 Prototypical Networks (1/5/10-shot)\n")

        model_paths = {}
        protonet_results = {}

        for shot in shot_list:
            print(f"\n--- {args.n_way}-way {shot}-shot ---")
            shot_config = build_config(args, shot)
            shot_config['tag'] = f'{args.backbone}_{args.n_way}way{shot}shot'
            model_path, acc = train(shot_config)
            model_paths[shot] = model_path

        print("\n" + "="*60)
        print("Phase 3/3: 微调基线 + 综合评估\n")

        finetune_results = {}
        _, _, test_classes, _, val_root = create_splits(args.data_root)

        for shot in shot_list:
            print(f"\n--- 测试 {args.n_way}-way {shot}-shot ---")

            # ProtoNet 评估
            shot_config = build_config(args, shot)
            res = evaluate_protonet(model_paths[shot], shot_config, shot_list=[shot])
            protonet_results[shot] = res[shot]

            # 微调评估
            acc, std = finetune_evaluate(shot_config, test_classes, val_root)

            ci95 = 1.96 * std / np.sqrt(shot_config['test_episodes'])
            finetune_results[shot] = (acc, ci95)

        # 最终汇总
        save_path = os.path.join(args.save_dir, 'comparison.png')
        plot_comparison(protonet_results, finetune_results, args.n_way, save_path)
        generate_report(protonet_results, finetune_results, config, args.save_dir)

        # 终端输出汇总表
        print(f"\n{'='*70}")
        print(f"  最终实验结果汇总 ({args.n_way}-way)")
        print(f"{'='*70}")
        print(f"  {'Setting':<15} {'ProtoNet':>18} {'Fine-tune':>18} {'Winner':>10}")
        print(f"  {'-'*65}")
        for shot in shot_list:
            p_acc = protonet_results[shot][0] * 100
            f_acc = finetune_results[shot][0] * 100
            winner = 'ProtoNet' if p_acc > f_acc else 'Fine-tune'
            print(f"  {shot}-shot{'':<8} {p_acc:>15.2f}% {f_acc:>15.2f}% {winner:>12}")
        print(f"{'='*70}")

    print(f"\n[✓] 任务完成！结果保存于: {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()
