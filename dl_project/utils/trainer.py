import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import calculate_metrics, plot_confusion_matrix


class Trainer:
    """训练器类"""

    def __init__(self, model, device, train_loader, test_loader, config, save_dir, logger=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_dir = save_dir
        self.logger = logger

        # 创建保存目录
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()

        optimizer_name = config.get('optimizer', 'SGD')  # 默认改为SGD
        lr = config.get('learning_rate', 0.1)  # 学习率改为0.1
        weight_decay = config.get('weight_decay', 5e-4)

        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':  # 添加AdamW优化器
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = config.get('momentum', 0.9)
            nesterov = config.get('nesterov', True)  # 添加Nesterov动量
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                       nesterov=nesterov, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 学习率调度器
        scheduler_name = config.get('scheduler', 'CosineAnnealingLR')
        self.scheduler_name = scheduler_name  # 保存调度器名称到实例变量

        if scheduler_name == 'CosineAnnealingLR':
            t_max = config.get('t_max', 50)
            eta_min = config.get('eta_min', 1e-6)  # 添加eta_min参数
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )
        elif scheduler_name == 'StepLR':
            step_size = config.get('step_size', 30)  # 调整步长
            gamma = config.get('gamma', 0.1)  # 调整gamma
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'MultiStepLR':
            milestones = config.get('milestones', [100, 150])
            gamma = config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            factor = config.get('factor', 0.1)
            patience = config.get('lr_patience', 10)
            min_lr = config.get('min_lr', 1e-6)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=factor,
                patience=patience, min_lr=min_lr, verbose=True
            )
        else:
            self.scheduler = None

        # 记录训练过程
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.lr_history = []
        self.best_acc = 0.0
        self.best_model_path = ""

        # 早停设置
        self.early_stopping = config.get('early_stopping', False)
        self.patience = config.get('patience', 10)
        self.counter = 0

        self.log("训练器初始化完成")
        self.log(f"优化器: {optimizer_name}, 学习率: {lr}")
        self.log(f"调度器: {scheduler_name}")

    def log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')

        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)

        return test_loss, test_acc, f1, all_preds, all_targets

    def train(self):
        """完整训练流程"""
        epochs = self.config.get('epochs', 50)
        save_best = self.config.get('save_best', True)
        save_frequency = self.config.get('save_frequency', 5)

        self.log(f"开始训练，共{epochs}个epoch")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 评估
            test_loss, test_acc, f1, _, _ = self.evaluate(epoch)

            # 学习率调整
            # 学习率调整
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)

                # 根据调度器类型调整调用方式
                if self.scheduler_name == 'ReduceLROnPlateau':
                    # ReduceLROnPlateau需要验证集指标
                    self.scheduler.step(test_acc)
                else:
                    self.scheduler.step()

            # 保存最佳模型
            if save_best and test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_path = os.path.join(
                    self.save_dir, 'checkpoints', f'best_model_epoch{epoch}_acc{test_acc:.2f}.pth'
                )
                # 精简保存：只保存模型和关键信息
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }, self.best_model_path)
                self.log(f"保存最佳模型，准确率: {test_acc:.2f}%")

            # 定期保存也精简
            if epoch % save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.save_dir, 'checkpoints', f'checkpoint_epoch{epoch}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, checkpoint_path)

            # 打印结果
            epoch_time = time.time() - start_time
            self.log(f"Epoch {epoch}/{epochs} | "
                     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                     f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                     f"F1 Score: {f1:.4f} | Time: {epoch_time:.1f}s")

            # 早停检查
            if self.early_stopping:
                if test_acc > self.best_acc:
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.log(f"早停触发，连续{self.patience}个epoch性能未提升")
                        break

        self.log(f"训练完成！最佳测试准确率: {self.best_acc:.2f}%")

        # 绘制训练曲线
        if self.config.get('plot_training_curves', True):
            self.plot_training_curves()

        return self.best_model_path

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.test_losses, label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Test Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(self.train_accs, label='Train Acc')
        axes[1].plot(self.test_accs, label='Test Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 学习率曲线
        if self.lr_history:
            axes[2].plot(self.lr_history)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'plots', 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"训练曲线已保存到: {plot_path}")

    def evaluate_final(self, model_path=None):
        """最终模型评估"""
        if model_path is None:
            model_path = self.best_model_path

        # 加载最佳模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 评估
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算指标
        metrics = calculate_metrics(all_targets, all_preds)

        # 绘制混淆矩阵 - 修复这里！
        if self.config.get('plot_confusion_matrix', True):
            # 定义CIFAR-10的类别名称
            cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck')

            plot_confusion_matrix(
                all_targets,
                all_preds,
                class_names=cifar10_classes,  # 传递类别名称
                save_path=os.path.join(self.save_dir, 'plots', 'confusion_matrix.png')
            )

        self.log("最终模型评估结果:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")

        return metrics


# trainer.py - 在LightweightTrainer类中添加优化方法
class LightweightTrainer(Trainer):
    """轻量级训练器（CPU优化）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CPU训练特别优化
        self.batch_losses = []  # 记录每个batch的loss

    def train_epoch(self, epoch):
        """训练一个epoch - CPU优化版本"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []

        # 使用简单进度条，减少开销
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            batch_losses.append(loss.item())

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 每50个batch打印一次，减少开销
            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        self.batch_losses.extend(batch_losses)  # 保存batch级loss

        return epoch_loss, epoch_acc
