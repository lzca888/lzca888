"""
调试测试脚本 - 定位问题
"""
import torch
import torch.nn as nn
from models.simple_resnet import SimpleResNet
from utils.data_loader import get_cifar10_dataloaders


def test_environment():
    print("=" * 60)
    print("1. 测试环境配置...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print("=" * 60)

    print("\n2. 测试数据加载...")
    try:
        train_loader, test_loader, class_names = get_cifar10_dataloaders({
            'batch_size': 4,  # 很小的batch用于测试
            'num_workers': 0,
            'data_dir': './data'
        })
        print(f"数据加载成功！")
        print(f"类别: {class_names}")

        # 取一个batch测试
        for inputs, targets in train_loader:
            print(f"输入形状: {inputs.shape}")  # 应该是 [4, 3, 32, 32]
            print(f"标签形状: {targets.shape}")  # 应该是 [4]
            print(f"标签值: {targets}")
            break
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False

    print("\n3. 测试模型创建...")
    try:
        model = SimpleResNet(num_classes=10, use_cbam=True)
        print(f"模型创建成功！")

        # 测试前向传播
        test_input = torch.randn(4, 3, 32, 32)  # 模拟一个batch
        output = model(test_input)
        print(f"模型输出形状: {output.shape}")  # 应该是 [4, 10]
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

    print("\n4. 测试完整训练步骤...")
    try:
        model = SimpleResNet(num_classes=10, use_cbam=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        print(f"前向传播成功，输出形状: {outputs.shape}")

        # 计算损失
        loss = criterion(outputs, targets)
        print(f"损失计算成功，损失值: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"反向传播成功")

    except Exception as e:
        print(f"训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("所有测试通过！可以开始训练。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_environment()
    if not success:
        print("\n请将上面的完整错误信息发给我，我会帮你解决。")