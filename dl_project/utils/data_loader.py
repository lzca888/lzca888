import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# data_loader.py - 修改get_cifar10_transforms函数，添加更强的数据增强
def get_cifar10_transforms(train_config, test_config):
    """获取CIFAR-10数据变换"""
    # 训练集变换 - 更强大的数据增强
    train_transforms = []

    # 基础增强
    if train_config.get('random_crop', True):
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if train_config.get('random_horizontal_flip', True):
        train_transforms.append(transforms.RandomHorizontalFlip())

    # 颜色增强
    if train_config.get('color_jitter', True):
        train_transforms.append(transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ))

    # 可选：Cutout正则化（需要额外的包，先注释）
    # if train_config.get('cutout', False):
    #     train_transforms.append(Cutout(n_holes=1, length=16))

    train_transforms.append(transforms.ToTensor())

    if train_config.get('normalize', True):
        train_transforms.append(transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ))

    transform_train = transforms.Compose(train_transforms)

    # 测试集变换保持不变
    test_transforms = [transforms.ToTensor()]

    if test_config.get('normalize', True):
        test_transforms.append(transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ))

    transform_test = transforms.Compose(test_transforms)

    return transform_train, transform_test


def get_cifar10_dataloaders(data_config):
    """获取CIFAR-10数据加载器"""
    transform_train, transform_test = get_cifar10_transforms(
        data_config.get('train_transform', {}),
        data_config.get('test_transform', {})
    )

    # 下载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_config.get('data_dir', './data'),
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_config.get('data_dir', './data'),
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    batch_size = data_config.get('batch_size', 128)
    num_workers = data_config.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # CIFAR-10类别名称
    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, class_names


# data_loader.py - 添加CIFAR-100支持
def get_cifar_dataloaders(data_config, dataset_name='cifar10'):
    """获取CIFAR数据加载器，支持CIFAR-10和CIFAR-100"""
    transform_train, transform_test = get_cifar10_transforms(
        data_config.get('train_transform', {}),
        data_config.get('test_transform', {})
    )

    # 选择数据集
    if dataset_name.lower() == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
        class_names = None  # CIFAR-100有100个类别
        print(f"使用数据集: CIFAR-100")
    else:
        dataset_class = torchvision.datasets.CIFAR10
        class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"使用数据集: CIFAR-10")

    # 下载数据集
    train_dataset = dataset_class(
        root=data_config.get('data_dir', './data'),
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = dataset_class(
        root=data_config.get('data_dir', './data'),
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    batch_size = data_config.get('batch_size', 128)
    num_workers = data_config.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 对于CIFAR-100，使用类别索引作为名称
    if dataset_name.lower() == 'cifar100' and class_names is None:
        class_names = [f'class_{i}' for i in range(100)]

    return train_loader, test_loader, class_names

def get_data_transforms(normalize=True, augment=True):
    """获取标准数据变换（兼容旧代码）"""
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return transform_train, transform_test