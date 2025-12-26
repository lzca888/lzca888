# complete_evaluation.py
"""
完整模型评估脚本 - 集成AblationStudy和ModelEvaluator
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入现有的类
sys.path.append('.')
from ablation_study import AblationStudy
from model_evaluation import ModelEvaluator


class CompleteModelEvaluation:
    """完整的模型评估类"""

    def __init__(self, models_config):
        self.models_config = models_config
        self.ablation_results = None
        self.evaluation_results = {}

    def run_ablation_study(self):
        """运行消融实验"""
        print("=" * 60)
        print("运行消融实验...")
        print("=" * 60)

        study = AblationStudy(self.models_config)
        self.ablation_results = study.run_study(save_dir='./ablation_results')

        # 转换格式
        for model_name, result in self.ablation_results.items():
            self.evaluation_results[model_name] = {
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'confusion_matrix': result['confusion_matrix'],
                'predictions': result['predictions'],
                'targets': result['targets']
            }

        return self.evaluation_results

    def generate_comprehensive_reports(self):
        """生成综合报告"""
        if not self.evaluation_results:
            print("没有评估结果，请先运行消融实验")
            return None

        print("\n" + "=" * 60)
        print("生成综合评估报告...")
        print("=" * 60)

        # 创建评估器
        evaluator = ModelEvaluator(self.models_config)
        evaluator.evaluation_results = self.evaluation_results

        # 生成报告
        evaluator.generate_detailed_reports(save_dir='./comprehensive_reports')
        summary = evaluator.create_summary_report(save_dir='./comprehensive_reports')

        # 生成额外分析
        self.generate_additional_analysis(summary)

        return summary

    def generate_additional_analysis(self, summary_df):
        """生成额外分析"""
        print("\n生成额外分析...")

        # 1. 计算CBAM带来的提升
        cbam_analysis = self.analyze_cbam_effect(summary_df)

        # 2. 生成对比图表
        self.plot_comparison_charts(summary_df)

        # 3. 生成模型复杂度分析
        self.analyze_model_complexity()

        return cbam_analysis

    def analyze_cbam_effect(self, summary_df):
        """分析CBAM效果"""
        print("分析CBAM注意力机制的效果...")

        analysis_data = []
        for dataset in ['CIFAR-10', 'CIFAR-100']:
            subset = summary_df[summary_df['Dataset'] == dataset]

            if len(subset) >= 2:
                # 尝试找到有CBAM和无CBAM的模型
                with_cbam = subset[subset['CBAM'] == 'Yes']
                without_cbam = subset[subset['CBAM'] == 'No']

                if len(with_cbam) > 0 and len(without_cbam) > 0:
                    cbam_acc = float(with_cbam['Accuracy'].iloc[0].replace('%', ''))
                    nocbam_acc = float(without_cbam['Accuracy'].iloc[0].replace('%', ''))

                    improvement = cbam_acc - nocbam_acc
                    relative_improvement = (improvement / nocbam_acc) * 100

                    analysis_data.append({
                        'Dataset': dataset,
                        'Without_CBAM_Acc': f'{nocbam_acc:.2f}%',
                        'With_CBAM_Acc': f'{cbam_acc:.2f}%',
                        'Absolute_Improvement': f'{improvement:.2f}%',
                        'Relative_Improvement': f'{relative_improvement:.2f}%'
                    })

        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            analysis_df.to_csv('./comprehensive_reports/cbam_effect_analysis.csv', index=False)
            print("CBAM效果分析已保存")
            return analysis_df

        return None

    def plot_comparison_charts(self, summary_df):
        """绘制对比图表"""
        print("绘制对比图表...")

        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. 准确率对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 为每个数据集绘制
        for idx, dataset in enumerate(['CIFAR-10', 'CIFAR-100']):
            subset = summary_df[summary_df['Dataset'] == dataset]

            if not subset.empty:
                ax = axes[idx] if len(axes) > 1 else axes

                x = range(len(subset))
                bars = ax.bar(x, subset['Accuracy'].str.replace('%', '').astype(float))

                # 根据是否使用CBAM设置颜色
                for i, (_, row) in enumerate(subset.iterrows()):
                    color = 'lightcoral' if row['CBAM'] == 'Yes' else 'skyblue'
                    bars[i].set_color(color)

                ax.set_title(f'{dataset} - 模型准确率对比')
                ax.set_xlabel('模型')
                ax.set_ylabel('准确率 (%)')
                ax.set_xticks(x)
                ax.set_xticklabels(subset['Model'], rotation=45, ha='right')
                ax.set_ylim([0, 100])

                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('./comprehensive_reports/accuracy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("对比图表已保存")

    def analyze_model_complexity(self):
        """分析模型复杂度"""
        print("分析模型复杂度...")

        # 这里可以计算参数量、FLOPs等
        # 暂时使用模拟数据
        complexity_data = [
            {'Model': 'ResNet18_NoCBAM', 'Parameters (M)': 11.2, 'FLOPs (G)': 1.8},
            {'Model': 'ResNet18_CBAM', 'Parameters (M)': 11.3, 'FLOPs (G)': 1.9},
        ]

        complexity_df = pd.DataFrame(complexity_data)
        complexity_df.to_csv('./comprehensive_reports/model_complexity.csv', index=False)

        return complexity_df

    def run(self):
        """运行完整的评估流程"""
        print("开始完整的模型评估流程")
        print("=" * 60)

        # 1. 运行消融实验
        self.run_ablation_study()

        # 2. 生成综合报告
        summary = self.generate_comprehensive_reports()

        print("\n" + "=" * 60)
        print("评估流程完成！")
        print(f"所有报告已保存到:")
        print(f"  消融实验结果: ./ablation_results/")
        print(f"  综合评估报告: ./comprehensive_reports/")
        print("=" * 60)

        return summary


def main():
    """主函数"""
    # 模型配置
    models_config = [
        {
            'name': 'ResNet18_withCBAM_CIFAR10',
            'path': './results/resnet18_cbam/checkpoints/best_model_epoch146_acc95.65.pth',
            'dataset': 'cifar10',
            'use_cbam': True
        },
        {
            'name': 'ResNet18_NoCBAM_CIFAR10',
            'path': './results/resnet18_baseline/checkpoints/best_model_epoch150_acc95.19.pth',
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

    # 创建评估对象并运行
    evaluator = CompleteModelEvaluation(models_config)
    results = evaluator.run()

    return results


if __name__ == "__main__":
    main()