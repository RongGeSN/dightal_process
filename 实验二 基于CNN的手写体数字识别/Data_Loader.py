# ==================== Data_Loader.py ====================
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np


def get_mnist_loaders(batch_size=64, custom_transform=None):
    """
    构建并返回MNIST数据加载器

    作用：下载（如果本地没有）MNIST数据集，应用预处理变换，
         并创建PyTorch的DataLoader对象，方便批量加载数据

    参数:
        batch_size (int): 批次大小，默认64。每个批次包含多少张图像
        custom_transform: 自定义的预处理变换，如果为None则不做特殊处理

    返回:
        train_loader (DataLoader): 训练数据加载器，用于遍历训练集
        test_loader (DataLoader): 测试数据加载器，用于遍历测试集
        train_dataset (Dataset): 原始训练数据集对象，用于保存样本等操作
    """

    # 定义数据存储路径
    # 所有下载的数据集文件会保存在当前目录下的data文件夹中
    data_path = 'data'

    # 下载并加载训练数据集
    # train=True表示加载训练集，download=True表示如果本地没有则自动下载
    train_dataset = datasets.MNIST(
        root=data_path,  # 数据存储根目录
        train=True,  # 加载训练集
        download=True,  # 自动下载
        transform=custom_transform  # 应用预处理变换
    )

    # 下载并加载测试数据集
    # train=False表示加载测试集
    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=custom_transform
    )

    # 创建DataLoader对象
    # DataLoader负责批量加载数据，支持多线程、打乱顺序等功能

    # 训练加载器：shuffle=True表示每个epoch随机打乱数据顺序
    # 这有助于防止模型记住数据顺序，提高泛化能力
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 每个批次的样本数
        shuffle=True  # 训练时打乱顺序
    )

    # 测试加载器：shuffle=False表示不打乱顺序
    # 测试时不需要打乱，保持有序方便评估
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,  # 测试时可以用更大的批次，因为不需要反向传播
        shuffle=False  # 测试时不需要打乱
    )

    # 打印数据集信息，方便确认数据已正确加载
    print(f"--- 数据集状态 ---")
    print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")

    # 返回三个对象
    return train_loader, test_loader, train_dataset


def save_processed_samples(dataset, save_dir="./data/processed_samples"):
    """
    将处理后的Tensor保存为真正的图片文件

    作用：将预处理后的张量数据导出为PNG图片文件，
         方便在实验报告中展示预处理效果

    参数:
        dataset: 数据集对象（如train_dataset），包含处理后的图像张量
        save_dir (str): 保存图片的目录路径，默认"./data/processed_samples"
    """

    # 如果保存目录不存在，则创建它
    # os.path.exists检查路径是否存在，os.makedirs递归创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在保存处理后的样本图片到: {save_dir} ...")

    # 导出前10张作为示例
    for i in range(10):
        # 获取第i张图像和对应的标签
        img_tensor, label = dataset[i]

        # 将Tensor转换为可在PIL中显示的数组
        # squeeze()去掉维度为1的维度，因为MNIST是单通道，形状从(1,28,28)变成(28,28)
        img_array = img_tensor.squeeze().numpy()

        # 线性映射到0-255范围
        # 原因：预处理后的张量可能经过标准化，值可能不在[0,1]范围内
        # 为了正确显示为灰度图，需要映射回0-255
        img_min = img_array.min()
        img_max = img_array.max()

        # 避免除以零：如果最大值等于最小值，所有像素值相同
        if img_max - img_min != 0:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255)
        else:
            img_array = np.zeros_like(img_array)  # 全黑图像

        # 创建PIL图像对象并保存
        # astype('uint8')：转换为0-255的无符号整数类型，PIL要求这种格式
        img = Image.fromarray(img_array.astype('uint8'))

        # 保存文件，文件名包含标签信息，方便查看
        img.save(f"{save_dir}/sample_{i}_label_{label}.png")

    print(f"保存完成！请查看 {save_dir} 文件夹。")


# ==================== 独立运行测试 ====================
if __name__ == "__main__":
    # 这段代码仅在直接运行此文件时执行，被导入时不执行

    # 尝试导入Preprocessing模块中的预处理函数
    try:
        from Preprocessing import my_transform
    except ImportError:
        # 如果Preprocessing.py不存在，使用默认变换（仅转换为张量）
        print("未找到 Preprocessing.py，将使用默认变换。")
        my_transform = transforms.ToTensor()

    # 1. 获取加载器和原始数据集对象
    # 使用自定义的预处理变换
    _, _, train_data = get_mnist_loaders(custom_transform=my_transform)

    # 2. 保存处理后的样本图片
    # 这样可以直接在本地文件夹中看到预处理后的图像效果
    save_processed_samples(train_data)