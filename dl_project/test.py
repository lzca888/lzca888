import torch
import torch.nn as nn
from models.simple_resnet import SimpleResNet
from utils.data_loader import get_cifar10_dataloaders


def test():
    # 设置设备
    device = torch.device("cpu")

    # 加载数据
    train_loader, test_loader, class_names = get_cifar10_dataloaders({
        'batch_size': 64,
        'num_workers': 0,
        'data_dir': './data'
    })

    # 创建模型
    model = SimpleResNet(num_classes=10, use_cbam=True).to(device)

    # 测试一个batch
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")

        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")

        loss = criterion(outputs, targets)
        print(f"Loss: {loss.item()}")

        # 反向传播
        loss.backward()
        print("Backward pass completed.")

        break  # 只测试一个batch


if __name__ == "__main__":
    test()