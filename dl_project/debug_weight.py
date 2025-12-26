# debug_weights.py
import torch

# 检查四个权重文件的结构
weight_files = [
    './results/resnet18_cbam/checkpoints/best_model_epoch148_acc95.30.pth',
    './results/resnet18_baseline/checkpoints/best_model_epoch147_acc95.18.pth',
    './results/resnet18_cifar100_cbam/checkpoints/best_model_epoch188_acc78.46.pth',
    './results/resnet18_cifar100_baseline/checkpoints/best_model_epoch194_acc78.70.pth'
]

for file_path in weight_files:
    print(f"\n检查文件: {file_path}")
    try:
        checkpoint = torch.load(file_path, map_location='cpu')

        # 如果是字典，检查键名
        if isinstance(checkpoint, dict):
            print(f"  字典键: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 检查是否有CBAM相关的键
        cbam_keys = [k for k in state_dict.keys() if 'cbam' in k.lower()]
        print(f"  CBAM相关键数量: {len(cbam_keys)}")
        if cbam_keys:
            print(f"  前3个CBAM键: {cbam_keys[:3]}")

        # 检查总键数
        print(f"  总键数: {len(state_dict.keys())}")

    except Exception as e:
        print(f"  加载失败: {e}")