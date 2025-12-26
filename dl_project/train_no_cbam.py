"""
训练无CBAM的完整ResNet模型（专门用于消融实验）
"""
import torch
import argparse
from models.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
from utils.data_loader import get_cifar10_dataloaders
from utils.trainer import LightweightTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--save_dir', type=str, default='./results_resnet_no_cbam')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cpu")
    print("=" * 60)
    print(f"训练无CBAM的完整{args.model_type}模型")
    print(f"设备: {device}")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader, class_names = get_cifar10_dataloaders({
        'batch_size': args.batch_size,
        'num_workers': 0,
        'data_dir': './data'
    })

    # 创建模型 - 关键：use_cbam=False
    if args.model_type == 'resnet18':
        model = resnet18_cbam(num_classes=10, use_cbam=False)
    elif args.model_type == 'resnet34':
        model = resnet34_cbam(num_classes=10, use_cbam=False)
    elif args.model_type == 'resnet50':
        model = resnet50_cbam(num_classes=10, use_cbam=False)

    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model_type} (无CBAM)")
    print(f"参数量: {total_params:,} (约{total_params / 1e6:.2f}M)")

    # 训练配置（保持和CBAM版本相同）
    # train_no_cbam.py - 修改训练配置部分
    config = {
        'epochs': args.epochs,
        'optimizer': 'SGD',  # ✅ 改为SGD，与CBAM版本一致
        'learning_rate': 0.1,  # ✅ 使用相同学习率
        'momentum': 0.9,  # ✅ 添加动量
        'weight_decay': 5e-4,  # ✅ 权重衰减
        'scheduler': 'CosineAnnealingLR',  # ✅ 使用相同调度器
        't_max': args.epochs,
        'eta_min': 0,
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