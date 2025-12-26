"""
优化版CPU训练脚本 - 针对AMD CPU特别优化
"""
import torch
import argparse
import os
from models.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
from utils.data_loader import get_cifar10_dataloaders
from utils.trainer import LightweightTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)  # 稍微多些，但有早停
    parser.add_argument('--batch_size', type=int, default=32)  # 减小batch size提高精度
    parser.add_argument('--use_cbam', action='store_true', default=True)
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--save_dir', type=str, default='./results_optimized')
    parser.add_argument('--learning_rate', type=float, default=0.05)  # 稍小的学习率
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cpu")

    # 设置CPU优化
    torch.set_num_threads(4)  # 根据CPU核心数调整
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(4)

    print("=" * 60)
    print(f"优化CPU训练配置:")
    print(f"  模型: {args.model_type} {'+ CBAM' if args.use_cbam else ''}")
    print(f"  设备: {device}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print("=" * 60)

    # 加载数据 - 添加数据增强
    train_loader, test_loader, class_names = get_cifar10_dataloaders({
        'batch_size': args.batch_size,
        'num_workers': 0,
        'data_dir': './data',
        'use_augmentation': True  # 假设你的数据加载器支持这个参数
    })

    # 创建模型
    if args.model_type == 'resnet18':
        model = resnet18_cbam(num_classes=10, use_cbam=args.use_cbam)
    elif args.model_type == 'resnet34':
        model = resnet34_cbam(num_classes=10, use_cbam=args.use_cbam)
    elif args.model_type == 'resnet50':
        model = resnet50_cbam(num_classes=10, use_cbam=args.use_cbam)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型信息:")
    print(f"  类型: {args.model_type}")
    print(f"  使用CBAM: {args.use_cbam}")
    print(f"  总参数量: {total_params:,} (约{total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,}")

    # 优化训练配置
    config = {
        'epochs': args.epochs,
        'optimizer': 'SGD',
        'learning_rate': args.learning_rate,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'StepLR',  # 改用StepLR，更适合CPU训练
        'step_size': 20,  # 每20个epoch减少学习率
        'gamma': 0.5,  # 每次减少为原来的一半
        'save_best': True,
        'plot_training_curves': True,
        'plot_confusion_matrix': True,
        'early_stopping': True,
        'patience': 15,  # 增加耐心值
        'save_frequency': 10
    }

    # 创建训练器
    trainer = LightweightTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        save_dir=args.save_dir,
        logger=None
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()