"""
消融实验脚本 - 比较CBAM在不同数据集上的效果
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from models.resnet_cbam import resnet18_cbam
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


class AblationStudy:
    def __init__(self, models_config, device='cuda'):
        """
        models_config: 模型配置列表
        [{'name': 'ResNet18_CBAM_CIFAR10', 'path': '...', 'dataset': 'cifar10'},
         {'name': 'ResNet18_NoCBAM_CIFAR10', 'path': '...', 'dataset': 'cifar10'},
         ...]
        """
        self.models_config = models_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def load_model(self, model_config):
        """加载训练好的模型"""
        print(f"加载模型: {model_config['name']}")

        # 根据数据集确定类别数
        if model_config['dataset'] == 'cifar100':
            num_classes = 100
        else:
            num_classes = 10

        # 根据模型名称判断是否使用CBAM
        model_name = model_config['name'].lower()
        if 'nocbam' in model_name or 'no_cbam' in model_name:
            use_cbam = False
        elif 'cbam' in model_name:
            use_cbam = True
        else:
            use_cbam = False

        # 关键修复：根据检查点的实际形状确定reduction
        # 加载检查点文件查看实际参数
        checkpoint = torch.load(model_config['path'], map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 自动推断reduction值
        reduction = 16  # 默认值

        # 寻找CBAM层的参数来推断reduction
        for key in state_dict.keys():
            if 'cbam' in key and 'channel_attention.fc.0.weight' in key:
                # 形状是 [out_channels, in_channels]
                out_channels, in_channels = state_dict[key].shape
                if in_channels > out_channels:  # 确保是合理的值
                    inferred_reduction = in_channels // out_channels
                    print(f"  从{key}推断reduction={inferred_reduction} ({in_channels}/{out_channels})")
                    reduction = inferred_reduction
                    break

        # 根据推断的reduction创建模型
        model = resnet18_cbam(num_classes=num_classes, use_cbam=use_cbam, reduction=reduction)
        model = model.to(self.device)

        # 加载权重
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model, use_cbam, num_classes

    def load_dataset(self, dataset_name):
        """加载数据集"""
        print(f"加载数据集: {dataset_name}")

        if dataset_name == 'cifar100':
            # CIFAR-100
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2762)
            dataset_class = torchvision.datasets.CIFAR100
        else:
            # CIFAR-10
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            dataset_class = torchvision.datasets.CIFAR10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_dataset = dataset_class(
            root='./data',
            train=False,
            download=False,
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=4
        )

        return test_loader, len(test_dataset.classes)

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_targets = []
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        accuracy = total_correct / total_samples

        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_preds, average='macro')

        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }

    def run_study(self, save_dir='./ablation_results'):
        """运行消融实验"""
        os.makedirs(save_dir, exist_ok=True)

        for config in self.models_config:
            print(f"\n{'=' * 50}")
            print(f"处理: {config['name']}")
            print(f"{'=' * 50}")

            # 加载模型
            model, use_cbam, num_classes = self.load_model(config)

            # 加载数据集
            test_loader, _ = self.load_dataset(config['dataset'])

            # 评估模型
            results = self.evaluate_model(model, test_loader)

            # 保存use_cbam信息
            results['use_cbam'] = use_cbam

            # 保存结果
            self.results[config['name']] = results

            # 打印结果
            print(f"准确率: {results['accuracy'] * 100:.2f}%")
            print(f"F1分数: {results['f1_score']:.4f}")
            print(f"使用CBAM: {use_cbam}")

            # 保存每个模型的详细结果
            self.save_model_results(config['name'], results, save_dir)

        # 生成对比报告
        self.generate_comparison_report(save_dir)

        return self.results

    def save_model_results(self, model_name, results, save_dir):
        """保存单个模型的结果"""
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # 保存评估指标
        metrics_df = pd.DataFrame({
            'accuracy': [results['accuracy']],
            'f1_score': [results['f1_score']]
        })
        metrics_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)

        # 保存预测结果
        pred_df = pd.DataFrame({
            'target': results['targets'],
            'prediction': results['predictions']
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)

        # 保存混淆矩阵
        cm_df = pd.DataFrame(results['confusion_matrix'])
        cm_df.to_csv(os.path.join(model_dir, 'confusion_matrix.csv'), index=False)

        # 绘制混淆矩阵
        self.plot_confusion_matrix(
            results['confusion_matrix'],
            os.path.join(model_dir, 'confusion_matrix.png')
        )

    def plot_confusion_matrix(self, cm, save_path, class_names=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))

        if class_names is None:
            # 根据混淆矩阵大小生成类别名称
            n_classes = cm.shape[0]
            if n_classes == 10:
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck']
            else:
                class_names = [f'Class {i}' for i in range(n_classes)]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if len(class_names) <= 20 else False,
                    yticklabels=class_names if len(class_names) <= 20 else False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def generate_comparison_report(self, save_dir):
        """生成对比报告"""
        print(f"\n{'=' * 60}")
        print("消融实验结果对比")
        print(f"{'=' * 60}")

        # 提取结果
        comparison_data = []
        for model_name, results in self.results.items():
            # 从results中获取use_cbam，而不是从名称判断
            if 'use_cbam' in results:
                has_cbam = results['use_cbam']
            else:
                # 如果results中没有use_cbam，使用更精确的名称判断
                model_name_lower = model_name.lower()
                if 'nocbam' in model_name_lower or 'without' in model_name_lower:
                    has_cbam = False
                elif 'cbam' in model_name_lower:
                    has_cbam = True
                else:
                    has_cbam = False  # 默认值

            dataset = 'CIFAR-100' if '100' in model_name else 'CIFAR-10'

            comparison_data.append({
                'Model': model_name,
                'Dataset': dataset,
                'CBAM': 'Yes' if has_cbam else 'No',
                'Accuracy (%)': results['accuracy'] * 100,
                'F1 Score': results['f1_score']
            })

        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        df = df.sort_values(['Dataset', 'CBAM'])

        # 打印调试信息
        print("\n数据检查:")
        print(df[['Model', 'Dataset', 'CBAM']])

        # 检查每个数据集是否有Yes和No
        for dataset in ['CIFAR-10', 'CIFAR-100']:
            df_dataset = df[df['Dataset'] == dataset]
            has_yes = (df_dataset['CBAM'] == 'Yes').any()
            has_no = (df_dataset['CBAM'] == 'No').any()
            print(f"\n{dataset}:")
            print(f"  有CBAM=Yes: {has_yes}")
            print(f"  有CBAM=No: {has_no}")
            if not (has_yes and has_no):
                print(f"  ⚠️ 警告: {dataset}缺少CBAM或NoCBAM的模型")

        # 保存对比表格
        df.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)

        # 打印表格
        print("\n模型对比:")
        print(df.to_string(index=False))

        # 绘制对比图
        self.plot_comparison_charts(df, save_dir)

        # 计算CBAM带来的提升
        self.calculate_cbam_improvement(df, save_dir)

    def plot_comparison_charts(self, df, save_dir):
        """绘制对比图表"""
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. 准确率对比柱状图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # CIFAR-10对比
        df_cifar10 = df[df['Dataset'] == 'CIFAR-10']
        if not df_cifar10.empty:
            ax1 = axes[0]
            x = range(len(df_cifar10))
            # 根据CBAM状态设置颜色
            colors = ['skyblue' if cbam == 'No' else 'lightcoral'
                      for cbam in df_cifar10['CBAM']]
            bars = ax1.bar(x, df_cifar10['Accuracy (%)'], color=colors)
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('CIFAR-10: Accuracy Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_cifar10['Model'], rotation=45, ha='right')
            ax1.bar_label(bars, fmt='%.2f%%', padding=3)
            ax1.set_ylim([0, 100])

        # CIFAR-100对比
        df_cifar100 = df[df['Dataset'] == 'CIFAR-100']
        if not df_cifar100.empty:
            ax2 = axes[1]
            x = range(len(df_cifar100))
            # 根据CBAM状态设置颜色
            colors = ['skyblue' if cbam == 'No' else 'lightcoral'
                      for cbam in df_cifar100['CBAM']]
            bars = ax2.bar(x, df_cifar100['Accuracy (%)'], color=colors)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('CIFAR-100: Accuracy Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df_cifar100['Model'], rotation=45, ha='right')
            ax2.bar_label(bars, fmt='%.2f%%', padding=3)
            ax2.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. CBAM效果对比（分组柱状图） - 修复这里！
        # 确保每个数据集都有Yes和No
        need_cbam_effect_chart = True

        for dataset in ['CIFAR-10', 'CIFAR-100']:
            df_dataset = df[df['Dataset'] == dataset]
            unique_cbam = df_dataset['CBAM'].unique()
            if len(unique_cbam) < 2:
                print(f"⚠️ 无法绘制CBAM效果对比图：{dataset}缺少CBAM或NoCBAM数据")
                print(f"  当前数据: {unique_cbam}")
                need_cbam_effect_chart = False
                break

        if need_cbam_effect_chart:
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(2)  # CIFAR-10 和 CIFAR-100
            width = 0.35

            # 安全地获取数据
            def get_accuracy(df_dataset, cbam_value):
                subset = df_dataset[df_dataset['CBAM'] == cbam_value]
                if not subset.empty:
                    return subset['Accuracy (%)'].values[0]
                return None

            # 获取有CBAM和无CBAM的准确率
            cifar10_nocbam = get_accuracy(df_cifar10, 'No')
            cifar10_cbam = get_accuracy(df_cifar10, 'Yes')
            cifar100_nocbam = get_accuracy(df_cifar100, 'No')
            cifar100_cbam = get_accuracy(df_cifar100, 'Yes')

            # 只绘制存在的数据
            rects1_data = []
            rects2_data = []
            datasets_to_show = []

            if cifar10_nocbam is not None and cifar10_cbam is not None:
                rects1_data.append(cifar10_nocbam)
                rects2_data.append(cifar10_cbam)
                datasets_to_show.append('CIFAR-10')

            if cifar100_nocbam is not None and cifar100_cbam is not None:
                rects1_data.append(cifar100_nocbam)
                rects2_data.append(cifar100_cbam)
                datasets_to_show.append('CIFAR-100')

            if rects1_data and rects2_data:
                x = np.arange(len(datasets_to_show))
                rects1 = ax.bar(x - width / 2, rects1_data,
                                width, label='Without CBAM', color='skyblue')
                rects2 = ax.bar(x + width / 2, rects2_data,
                                width, label='With CBAM', color='lightcoral')

                ax.set_xlabel('Dataset')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Effect of CBAM on Different Datasets')
                ax.set_xticks(x)
                ax.set_xticklabels(datasets_to_show)
                ax.legend()
                ax.bar_label(rects1, fmt='%.2f%%', padding=3)
                ax.bar_label(rects2, fmt='%.2f%%', padding=3)
                ax.set_ylim([0, 100])

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'cbam_effect_comparison.png'),
                            dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print("⚠️ 无法绘制CBAM效果对比图：缺少必要的数据")

        # 3. 训练曲线对比
        self.plot_training_curves_comparison(save_dir)

    def plot_training_curves_comparison(self, save_dir):
        """绘制训练曲线对比（如果训练日志存在）"""
        # 尝试从训练日志中提取训练曲线
        training_logs_dir = './results'

        if os.path.exists(training_logs_dir):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for i, model_name in enumerate(self.results.keys()):
                if i >= 4:  # 最多显示4个模型的曲线
                    break

                # 寻找训练曲线图
                model_results_dir = os.path.join(training_logs_dir,
                                                 model_name.replace(' ', '_').lower())
                curves_path = os.path.join(model_results_dir, 'plots', 'training_curves.png')

                if os.path.exists(curves_path):
                    img = plt.imread(curves_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    axes[i].set_title(model_name)
                else:
                    axes[i].text(0.5, 0.5, f"No training curves\nfor {model_name}",
                                 ha='center', va='center')
                    axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves_comparison.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def calculate_cbam_improvement(self, df, save_dir):
        """计算CBAM带来的提升"""
        improvement_data = []

        for dataset in ['CIFAR-10', 'CIFAR-100']:
            df_dataset = df[df['Dataset'] == dataset]
            if len(df_dataset) == 2:  # 有CBAM和无CBAM都有
                cbam_acc = df_dataset[df_dataset['CBAM'] == 'Yes']['Accuracy (%)'].values[0]
                nocbam_acc = df_dataset[df_dataset['CBAM'] == 'No']['Accuracy (%)'].values[0]

                absolute_improvement = cbam_acc - nocbam_acc
                relative_improvement = (cbam_acc - nocbam_acc) / nocbam_acc * 100

                improvement_data.append({
                    'Dataset': dataset,
                    'Without CBAM (%)': nocbam_acc,
                    'With CBAM (%)': cbam_acc,
                    'Absolute Improvement (pp)': absolute_improvement,
                    'Relative Improvement (%)': relative_improvement
                })

        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            print("\n" + "=" * 60)
            print("CBAM效果提升分析")
            print("=" * 60)
            print(improvement_df.to_string(index=False))

            # 保存提升分析
            improvement_df.to_csv(os.path.join(save_dir, 'cbam_improvement_analysis.csv'), index=False)

            # 绘制提升效果图
            self.plot_improvement_chart(improvement_df, save_dir)

    def plot_improvement_chart(self, df, save_dir):
        """绘制提升效果图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        width = 0.35

        rects1 = ax.bar(x - width / 2, df['Without CBAM (%)'], width,
                        label='Without CBAM', color='skyblue')
        rects2 = ax.bar(x + width / 2, df['With CBAM (%)'], width,
                        label='With CBAM', color='lightcoral')

        # 添加提升标注
        for i, row in df.iterrows():
            improvement = row['Absolute Improvement (pp)']
            ax.text(i, max(row['Without CBAM (%)'], row['With CBAM (%)']) + 1,
                    f'+{improvement:.2f}%', ha='center', fontweight='bold')

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Improvement of CBAM Attention Mechanism')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Dataset'])
        ax.legend(loc='lower right')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cbam_improvement_chart.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """主函数 - 配置并运行消融实验"""
    # 配置你的模型路径
    models_config = [
        {
            'name': 'ResNet18_withCBAM_CIFAR10',
            'path': './results/resnet18_cbam/checkpoints/best_model_epoch146_acc95.65.pth',  # 修改为实际路径
            'dataset': 'cifar10',
            'use_cbam': True
        },
        {
            'name': 'ResNet18_NoCBAM_CIFAR10',
            'path': './results/resnet18_baseline/checkpoints/best_model_epoch150_acc95.19.pth',  # 修改为实际路径
            'dataset': 'cifar10',
            'use_cbam': False
        },
        {
            'name': 'ResNet18_NoCBAM_CIFAR100',
            'path': './results/resnet18_cifar100_no_cbam/checkpoints/best_model_epoch190_acc78.39.pth',
            'dataset': 'cifar100',
            'use_cbam': False
        },
        {
            'name': 'ResNet18_withCBAM_CIFAR100',
            'path': './results/resnet18_cifar100_cbam/checkpoints/best_model_epoch193_acc78.91.pth',
            'dataset': 'cifar100',
            'use_cbam': True
        }
    ]

    # 创建消融实验对象
    study = AblationStudy(models_config)

    # 运行消融实验
    results = study.run_study(save_dir='./ablation_results')

    print("\n" + "=" * 60)
    print("消融实验完成！")
    print(f"结果已保存到: ./ablation_results/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()