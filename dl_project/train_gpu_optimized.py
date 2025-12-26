"""
GPU优化训练脚本 - 专为RTX 4090优化
"""
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from models.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
from utils.data_loader import get_cifar10_dataloaders
from utils.trainer import Trainer
from utils.logger import setup_logger


class GPUTrainer(Trainer):
    """GPU优化训练器，支持混合精度训练和更高级的优化"""

    def __init__(self, model, device, train_loader, test_loader, config, save_dir, logger=None):
        super().__init__(model, device, train_loader, test_loader, config, save_dir, logger)

        # 混合精度训练
        self.scaler = GradScaler() if config.get('use_amp', True) else None

        # 标签平滑
        self.label_smoothing = config.get('label_smoothing', 0.1)

        # 学习率warmup
        self.warmup_epochs = config.get('warmup_epochs', 5)

        # 梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        self.log(f"GPU优化训练器初始化完成")
        if self.scaler:
            self.log("使用混合精度训练 (AMP)")
        if self.label_smoothing > 0:
            self.log(f"使用标签平滑: {self.label_smoothing}")

    def train_epoch(self, epoch):
        """训练一个epoch - GPU优化版本"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        accumulation_steps = 0

        # 学习率warmup
        if epoch <= self.warmup_epochs:
            warmup_lr = self.config['learning_rate'] * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 混合精度训练
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs)

                # 标签平滑的交叉熵损失
                if self.label_smoothing > 0:
                    loss = self.label_smoothing_loss(outputs, targets, self.label_smoothing)
                else:
                    loss = self.criterion(outputs, targets)

                # 梯度累积
                loss = loss / self.gradient_accumulation_steps

            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1

            # 梯度累积步骤
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_steps = 0

            # 统计
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def label_smoothing_loss(self, outputs, targets, smoothing=0.1):
        """标签平滑损失函数"""
        num_classes = outputs.size(-1)
        log_preds = torch.nn.functional.log_softmax(outputs, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)

        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)  # 增加epoch数
    parser.add_argument('--batch_size', type=int, default=256)  # 增加batch size
    parser.add_argument('--use_cbam', action='store_true', default=True)
    parser.add_argument('--model_type', type=str, default='resnet50',  # 使用更深的模型
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--save_dir', type=str, default='./results_gpu_optimized')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 启用cuDNN自动优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print("=" * 60)
    print(f"GPU优化训练配置:")
    print(f"  模型: {args.model_type} {'+ CBAM' if args.use_cbam else ''}")
    print(f"  设备: {device}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  标签平滑: {args.label_smoothing}")
    print("=" * 60)

    # 加载数据 - 增强版数据增强
    train_loader, test_loader, class_names = get_cifar10_dataloaders({
        'batch_size': args.batch_size,
        'num_workers': 8,  # 增加数据加载线程
        'data_dir': './data',
        'pin_memory': True,
        'train_transform': {
            'random_crop': True,
            'random_horizontal_flip': True,
            'color_jitter': True,  # 启用颜色增强
            'normalize': True
        },
        'test_transform': {
            'normalize': True
        }
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

    # 并行计算（如果有多GPU）
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型信息:")
    print(f"  类型: {args.model_type}")
    print(f"  使用CBAM: {args.use_cbam}")
    print(f"  总参数量: {total_params:,} (约{total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,}")

    # GPU优化训练配置
    config = {
        'epochs': args.epochs,
        'optimizer': 'SGD',
        'learning_rate': args.learning_rate,
        'momentum': 0.9,
        'weight_decay': 5e-4,  # 增加权重衰减
        'scheduler': 'CosineAnnealingLR',
        't_max': args.epochs - args.warmup_epochs,  # 余弦退火
        'eta_min': 1e-6,  # 更小的最小学习率
        'save_best': True,
        'plot_training_curves': True,
        'plot_confusion_matrix': True,
        'early_stopping': True,
        'patience': 30,  # 增加耐心值
        'save_frequency': 10,
        'use_amp': True,  # 启用混合精度
        'label_smoothing': args.label_smoothing,
        'warmup_epochs': args.warmup_epochs,
        'gradient_accumulation_steps': 1
    }

    # 设置日志
    logger = setup_logger(args.save_dir)

    # 创建GPU优化训练器
    trainer = GPUTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        save_dir=args.save_dir,
        logger=logger
    )

    # 开始训练
    trainer.train()

    # 最终评估
    print("\n最终模型评估...")
    final_metrics = trainer.evaluate_final()

    print("\n" + "=" * 60)
    print(f"实验完成！最佳准确率: {trainer.best_acc:.2f}%")
    print("最终评估结果:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()