import torch

# 加载检查点
checkpoint = torch.load('./results/resnet18_cifar100_cbam/checkpoints/best_model_epoch193_acc78.91.pth')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# 打印CBAM相关层的形状
for key, value in state_dict.items():
    if 'cbam' in key.lower() and 'weight' in key:
        print(f"{key}: {value.shape}")