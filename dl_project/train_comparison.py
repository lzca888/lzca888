"""
对比实验脚本 - 在RTX 4090上运行多个模型对比
"""
import subprocess
import time

# 实验配置
experiments = [
    {
        'name': 'ResNet18_No_CBAM',
        'model': 'resnet18',
        'use_cbam': False,
        'epochs': 150,
        'batch_size': 256,
        'lr': 0.1
    },
    {
        'name': 'ResNet18_CBAM',
        'model': 'resnet18',
        'use_cbam': True,
        'epochs': 150,
        'batch_size': 256,
        'lr': 0.1
    },
    {
        'name': 'ResNet34_CBAM',
        'model': 'resnet34',
        'use_cbam': True,
        'epochs': 150,
        'batch_size': 256,
        'lr': 0.1
    },
    {
        'name': 'ResNet50_CBAM',
        'model': 'resnet50',
        'use_cbam': True,
        'epochs': 200,
        'batch_size': 128,  # ResNet50需要更多显存
        'lr': 0.1
    }
]


def run_experiment(exp_config):
    """运行单个实验"""
    print(f"\n{'=' * 60}")
    print(f"开始实验: {exp_config['name']}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # 构建命令行参数
    cmd = [
        'python', 'train_gpu_optimized.py',
        '--model_type', exp_config['model'],
        '--epochs', str(exp_config['epochs']),
        '--batch_size', str(exp_config['batch_size']),
        '--learning_rate', str(exp_config['lr']),
        '--save_dir', f"./results_comparison/{exp_config['name']}"
    ]

    if not exp_config['use_cbam']:
        cmd.append('--no-use_cbam')

    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 记录结果
    end_time = time.time()
    duration = end_time - start_time

    print(f"实验 {exp_config['name']} 完成")
    print(f"耗时: {duration / 60:.2f} 分钟")
    print(f"输出: {result.stdout[-500:]}")  # 打印最后500个字符

    if result.returncode != 0:
        print(f"错误: {result.stderr}")

    return result.returncode


def main():
    print("开始对比实验...")
    print(f"GPU: RTX 4090")
    print(f"总计 {len(experiments)} 个实验")

    results = []
    for exp in experiments:
        return_code = run_experiment(exp)
        results.append({
            'name': exp['name'],
            'success': return_code == 0
        })

    print(f"\n{'=' * 60}")
    print("所有实验完成!")
    print("实验结果汇总:")
    for res in results:
        status = "✓ 成功" if res['success'] else "✗ 失败"
        print(f"  {res['name']}: {status}")


if __name__ == "__main__":
    main()