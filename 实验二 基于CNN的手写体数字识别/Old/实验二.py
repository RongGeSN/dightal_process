import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 数据集构建与预处理 (归一化 + 形态学处理)
# ==========================================

def morphology_processing(img):
    """
    模拟实验要求中的形态学处理（如：膨胀、腐蚀），增加模型鲁棒性
    这里在数据读取阶段通过 OpenCV 模拟处理
    """
    img_np = np.array(img)
    kernel = np.ones((2, 2), np.uint8)
    # 随机进行开运算，去除细小噪声
    img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
    return img_np


# 定义变换：1. 统一尺寸(28x28) 2. 转为张量 3. 归一化 (均值0.1307, 标准差0.3081)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Lambda(lambda x: morphology_processing(x)),  # 调用形态学处理
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# ==========================================
# 2. 搭建 CNN 网络 (包含激活函数与结构设计)
# ==========================================


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # 第一层卷积：输入1通道，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 归一化层，加速收敛

        # 第二层卷积：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 池化层：2x2 最大池化
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        # 经过两次池化，28x28 变为 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 最终输出10个类别(0-9)

    def forward(self, x):
        # 卷积 -> BN -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接 -> ReLU -> Dropout -> 输出
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 训练与识别
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN().to(device)

# 定义损失函数 (交叉熵) 和 优化器 (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()  # 梯度清零
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batch_idx % 200 == 0:
                print(f'Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                      f'\tLoss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 测试阶段不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 找概率最大的指数
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


if __name__ == '__main__':
    train(3)  # 训练3个轮次
    test()  # 执行识别测试