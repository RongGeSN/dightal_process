import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64, custom_transform=None):
    """
    构建 MNIST 数据集加载器
    """
    # 如果没有传入自定义预处理，则使用基础变换
    if custom_transform is None:
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # 加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=custom_transform)
    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=custom_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"数据集构建成功：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张。")
    return train_loader, test_loader