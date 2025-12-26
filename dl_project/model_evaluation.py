"""
综合模型评估和可视化脚本
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
import os

warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.evaluation_results = {}

    def load_and_evaluate(self):
        """加载并评估所有模型"""
        for config in self.model_configs:
            print(f"\n评估模型: {config['name']}")
            print("-" * 40)
            pass

    def generate_detailed_reports(self, save_dir='./evaluation_reports'):
        """生成详细评估报告"""
        os.makedirs(save_dir, exist_ok=True)

        # 生成分类报告
        for model_name, results in self.evaluation_results.items():
            print(f"为模型 {model_name} 生成报告...")

            # 确保数据是NumPy数组
            y_true = np.array(results['targets'])
            y_pred = np.array(results['predictions'])

            # 生成分类报告
            try:
                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )

                # 转换为DataFrame
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(save_dir, f'{model_name}_classification_report.csv'))

                # 绘制精度-召回率曲线
                self.plot_precision_recall_curve(results, model_name, save_dir)

                # 绘制ROC曲线
                self.plot_roc_curve(results, model_name, save_dir)

            except Exception as e:
                print(f"为模型 {model_name} 生成报告时出错: {e}")

    def plot_precision_recall_curve(self, results, model_name, save_dir):
        """绘制精度-召回率曲线"""
        # 为每个类别计算精度-召回率
        from sklearn.metrics import precision_recall_curve

        # 确保 y_true 和 y_pred 是NumPy数组
        y_true = np.array(results['targets'])
        y_pred = np.array(results['predictions'])

        n_classes = len(np.unique(y_true))

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i in range(min(n_classes, 10)):  # 最多显示10个类别
            # 二值化当前类别
            y_true_binary = (y_true == i).astype(int)
            y_scores = np.array([1 if p == i else 0 for p in y_pred])

            # 检查是否有数据
            if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true_binary):
                try:
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)

                    # 确保有足够的数据点
                    if len(precision) > 0 and len(recall) > 0:
                        axes[i].plot(recall, precision, marker='.', label=f'Class {i}')
                        axes[i].set_xlabel('Recall')
                        axes[i].set_ylabel('Precision')
                        axes[i].set_title(f'Class {i}')
                        axes[i].grid(True, alpha=0.3)
                    else:
                        axes[i].text(0.5, 0.5, f'No data for class {i}',
                                    ha='center', va='center')
                        axes[i].set_title(f'Class {i}')
                except:
                    axes[i].text(0.5, 0.5, f'Error for class {i}',
                                ha='center', va='center')
                    axes[i].set_title(f'Class {i}')
            else:
                axes[i].text(0.5, 0.5, f'No data for class {i}',
                            ha='center', va='center')
                axes[i].set_title(f'Class {i}')

        plt.suptitle(f'Precision-Recall Curves - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_precision_recall.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, results, model_name, save_dir):
        """绘制ROC曲线（适用于二分类或少量多分类）"""
        # 确保数据是NumPy数组
        y_true = np.array(results['targets'])
        y_pred = np.array(results['predictions'])
        n_classes = len(np.unique(y_true))

        if n_classes <= 10 and n_classes > 1:
            # 二值化标签
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            # 计算ROC曲线和AUC
            from sklearn.metrics import roc_curve, auc
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                # 为当前类别创建分数
                y_scores = np.array([1 if p == i else 0 for p in y_pred])

                # 确保有正负样本
                if np.sum(y_true_bin[:, i]) > 0 and np.sum(y_true_bin[:, i]) < len(y_true):
                    try:
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores)
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    except:
                        print(f"  警告: 类别 {i} 计算ROC曲线时出错")
                        continue
                else:
                    continue

            # 绘制所有类别的ROC曲线
            if fpr and tpr:
                plt.figure(figsize=(10, 8))
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green',
                                'purple', 'brown', 'pink', 'gray', 'olive'])

                for i, color in zip(range(n_classes), colors):
                    if i in fpr and i in tpr and i in roc_auc:
                        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curves.png'),
                            dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print(f"  无法为模型 {model_name} 绘制ROC曲线")
        else:
            print(f"  跳过ROC曲线绘制: 类别数 {n_classes} 过多或为1")

    def create_summary_report(self, save_dir='./evaluation_reports'):
        """创建总结报告"""
        import os

        summary_data = []

        for model_name, results in self.evaluation_results.items():
            # 解析模型信息
            dataset = 'CIFAR-100' if '100' in model_name else 'CIFAR-10'
            cbam = 'Yes' if 'CBAM' in model_name else 'No'

            # 计算各类指标
            from sklearn.metrics import precision_score, recall_score, f1_score

            # 确保数据是NumPy数组
            y_true = np.array(results['targets'])
            y_pred = np.array(results['predictions'])

            accuracy = results['accuracy']

            # 计算宏平均指标
            try:
                precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            except Exception as e:
                print(f"计算模型 {model_name} 指标时出错: {e}")
                precision_macro = 0
                recall_macro = 0
                f1_macro = 0

            summary_data.append({
                'Model': model_name,
                'Dataset': dataset,
                'CBAM': cbam,
                'Accuracy': f'{accuracy*100:.2f}%',
                'Accuracy_Value': accuracy,
                'Precision (Macro)': f'{precision_macro:.4f}',
                'Recall (Macro)': f'{recall_macro:.4f}',
                'F1-Score (Macro)': f'{f1_macro:.4f}'
            })

        # 创建总结表格
        summary_df = pd.DataFrame(summary_data)
        os.makedirs(save_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(save_dir, 'model_summary.csv'), index=False)

        # 生成LaTeX表格（便于插入报告）
        self.generate_latex_table(summary_df, save_dir)

        return summary_df

    def generate_latex_table(self, df, save_dir):
        """生成LaTeX格式的表格"""
        latex_table = df.to_latex(index=False,
                                  caption='Model Performance Comparison',
                                  label='tab:model_comparison')

        with open(os.path.join(save_dir, 'model_comparison.tex'), 'w') as f:
            f.write(latex_table)

        print("LaTeX表格已生成: model_comparison.tex")


# 添加主函数以便直接运行
if __name__ == "__main__":
    print("ModelEvaluator类定义文件")
    print("请使用complete_evaluation.py运行完整评估流程")