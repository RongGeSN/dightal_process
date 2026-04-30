import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

# 导入前两部分的模块
from Preprocessing import my_transform
from Data_Loader import get_mnist_loaders


# ==========================================
# 1. 搭建 CNN 网络 (SmartMnistCNN)
# ==========================================
class SmartMnistCNN(nn.Module):
    def __init__(self):
        super(SmartMnistCNN, self).__init__()
        # 卷积层部分：增加 BatchNorm 加速收敛，增加 Dropout 防止过拟合
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Dropout(0.25)
        )

        # 全连接层部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 最终输出10个分类
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# ==========================================
# 2. 训练、评估与随机可视化函数
# ==========================================

def train_and_evaluate():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # A. 获取数据加载器
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=64, custom_transform=my_transform)

    # B. 初始化模型、优化器和损失函数
    model = SmartMnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 用于记录绘图数据
    history = {'train_loss': [], 'test_acc': []}

    # C. 执行训练循环
    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 每个 Epoch 结束后计算测试集准确率
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                correct += preds.eq(target).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        history['train_loss'].append(running_loss / len(train_loader))
        history['test_acc'].append(acc)
        print(f"==> Epoch {epoch} 结束，测试集准确率: {acc:.2f}%")

    # D. 绘制训练指标曲线
    plot_training_curves(history)

    # E. 随机抽取测试样本进行 Debug 可视化
    visualize_random_predictions(model, device, test_loader)


def plot_training_curves(history):
    """绘制损失和准确率变化图"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], '-o', label='Train Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], '-s', color='orange', label='Test Accuracy')
    plt.title("Test Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_random_predictions(model, device, test_loader, num_samples=10):
    """随机抽取样本并展示 P(预测) 和 T(真实) 结果"""
    model.eval()
    dataset = test_loader.dataset
    # 随机产生不重复的索引
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(18, 4))

    # 在画布左侧添加说明
    info_text = "Legend:\nGreen: Correct\nRed: Incorrect\nT: True Label\nP: Predicted"
    plt.gcf().text(0.01, 0.5, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            # 准备输入数据 (Add batch dimension)
            input_data = img_tensor.unsqueeze(0).to(device)
            output = model(input_data)
            pred = output.argmax(dim=1).item()

            # 绘图设置
            ax = plt.subplot(1, num_samples, i + 1)
            # 反归一化显示 (近似处理)
            img_show = img_tensor.squeeze().numpy()
            plt.imshow(img_show, cmap='gray')

            color = 'green' if pred == label else 'red'
            ax.set_title(f"T: {label}\nP: {pred}", color=color, fontsize=14, fontweight='bold')
            plt.axis('off')

    plt.suptitle("CNN Recognition Results (Random Samples from Test Set)", fontsize=16, y=1.08)
    plt.show()


# ==================== 程序入口 ====================
if __name__ == "__main__":
    train_and_evaluate()