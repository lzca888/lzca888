import os

import pandas as pd


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