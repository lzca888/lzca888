"""
深度学习课程报告主训练脚本
支持CPU/GPU训练，配置化训练流程
"""
import yaml
import argparse
import torch
# 导入所有支持的模型
from models.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
from utils.data_loader import get_cifar10_dataloaders
from utils.trainer import Trainer
from utils.logger import setup_logger


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_config):
    """设置训练设备"""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)

    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def main(config_path):
    """主训练函数"""
    # 加载配置
    config = load_config(config_path)
    print("=" * 60)
    print(f"实验名称: {config['experiment']['name']}")
    print("=" * 60)

    # 设置设备
    device = setup_device(config['experiment']['device'])

    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    if device.type == "cuda":
        torch.cuda.manual_seed(config['experiment']['seed'])

    # 设置日志
    logger = setup_logger(config['output']['save_dir'])

    # 加载数据
    print("加载数据集...")
    data_config = config['data']
    dataset_name = data_config.get('name', 'cifar10')  # 默认为cifar10

    # 使用新的数据加载函数
    from utils.data_loader import get_cifar_dataloaders
    train_loader, test_loader, class_names = get_cifar_dataloaders(data_config, dataset_name)

    print(f"数据集: {dataset_name}, 类别数: {len(class_names)}")

    # 创建模型
    print("创建模型...")
    model_config = config['model']
    model_name = model_config['name']

    # 支持多种ResNet模型
    if model_name == "ResNet18_CBAM":
        model = resnet18_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=model_config['use_cbam'],
            reduction=model_config.get('cbam_reduction', 16)
        )
    elif model_name == "ResNet34_CBAM":
        model = resnet34_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=model_config['use_cbam'],
            reduction=model_config.get('cbam_reduction', 16)
        )
    elif model_name == "ResNet50_CBAM":
        model = resnet50_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=model_config['use_cbam'],
            reduction=model_config.get('cbam_reduction', 16)
        )
    elif model_name == "ResNet18":
        model = resnet18_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=False,
            reduction=model_config.get('cbam_reduction', 16)
        )
    elif model_name == "ResNet34":
        model = resnet34_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=False,
            reduction=model_config.get('cbam_reduction', 16)
        )
    elif model_name == "ResNet50":
        model = resnet50_cbam(
            num_classes=model_config['num_classes'],
            use_cbam=False,
            reduction=model_config.get('cbam_reduction', 16)
        )
    else:
        raise ValueError(f"不支持的模型: {model_config['name']}")

    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {model_config['name']}")
    print(f"使用CBAM: {model_config['use_cbam']}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config['training'],
        save_dir=config['output']['save_dir'],
        logger=logger
    )

    # 开始训练
    print("\n开始训练...")
    best_model_path = trainer.train()

    # 最终评估
    print("\n最终模型评估...")
    final_metrics = trainer.evaluate_final(best_model_path)

    print("\n" + "=" * 60)
    print(f"实验完成！最佳准确率: {final_metrics['accuracy']*100:.2f}%")
    print(f"模型已保存到: {best_model_path}")
    print("=" * 60)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深度学习课程报告训练脚本")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    main(args.config)