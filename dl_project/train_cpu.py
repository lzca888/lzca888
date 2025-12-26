"""
CPU优化训练脚本（使用完整ResNet模型）
"""
import torch
import argparse
# 修改这里：从models.resnet_cbam导入完整ResNet
from models.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
from utils.data_loader import get_cifar10_dataloaders
from utils.trainer import LightweightTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_cbam', action='store_true', default=True)
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--save_dir', type=str, default='./results_cpu')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cpu")
    print("=" * 60)
    print(f"训练配置:")
    print(f"  模型: {args.model_type} {'+ CBAM' if args.use_cbam else ''}")
    print(f"  设备: {device}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader, class_names = get_cifar10_dataloaders({
        'batch_size': args.batch_size,
        'num_workers': 0,  # CPU训练避免多进程
        'data_dir': './data'
    })

    # 创建模型（使用完整ResNet）
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

    # 训练配置
    # 训练配置 - 针对CBAM调整
    config = {
        'epochs': args.epochs,
        'optimizer': 'SGD',  # ✅ 改为SGD，更适合ResNet
        'learning_rate': 0.1,  # ✅ 增大学习率（CIFAR-10常用）
        'momentum': 0.9,  # ✅ 添加动量
        'weight_decay': 5e-4,  # ✅ 权重衰减
        'scheduler': 'CosineAnnealingLR',  # ✅ 改为余弦退火
        't_max': args.epochs,  # ✅ 余弦退火参数
        'eta_min': 0,  # ✅ 最小学习率
        'save_best': True,
        'plot_training_curves': True,
        'plot_confusion_matrix': True
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