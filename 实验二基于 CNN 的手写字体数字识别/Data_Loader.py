import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np


def get_mnist_loaders(batch_size=64, custom_transform=None):
    """
    构建并返回 MNIST 数据加载器
    """
    data_path = './data'

    # 下载并加载数据集
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=custom_transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=custom_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"--- 数据集状态 ---")
    print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset


def save_processed_samples(dataset, save_dir="./data/processed_samples"):
    """
    将处理后的 Tensor 存为真正的图片文件，方便实验报告引用
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在保存处理后的样本图片到: {save_dir} ...")
    for i in range(10):  # 导出前10张作为示例
        img_tensor, label = dataset[i]

        # 将 Tensor 还原为像素值 (0-255)
        # 注意：如果之前做了 Normalize，这里的还原可能需要根据均值标准差反推，
        # 但为了直观查看形态学效果，我们直接映射回 0-255。
        img_array = img_tensor.squeeze().numpy()
        # 线性映射到 0-255 范围
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255)

        img = Image.fromarray(img_array.astype('uint8'))
        img.save(f"{save_dir}/sample_{i}_label_{label}.png")
    print(f"保存完成！请查看 {save_dir} 文件夹。")


# ==================== 独立运行测试 ====================
if __name__ == "__main__":
    # 为了演示，这里导入第二部分的预处理逻辑
    try:
        from Preprocessing import my_transform
    except ImportError:
        print("未找到 Preprocessing.py，将使用默认变换。")
        my_transform = transforms.ToTensor()

    # 1. 获取加载器和原始数据集对象
    _, _, train_data = get_mnist_loaders(custom_transform=my_transform)

    # 2. 保存处理后的样本图片
    save_processed_samples(train_data)