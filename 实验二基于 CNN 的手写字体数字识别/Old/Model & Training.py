import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 注意：运行前确保你的 Preprocessing.py 和 Data_Loader.py 在同一目录下
from Preprocessing import my_transform
from Data_Loader import get_mnist_loaders


# 1. 搭建 CNN 网络 (保持不变)
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output


# 2. 训练函数（增加记录 loss 的逻辑）
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')
    return epoch_loss / len(train_loader)


# 3. 可视化函数：展示预测效果
def visualize_predictions(model, device, test_loader, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))  # 获取一组数据
    images, labels = images[:num_images].to(device), labels[:num_images]

    with torch.no_grad():
        output = model(images)
        preds = output.argmax(dim=1, keepdim=True)

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # 反归一化显示图像
        img = images[i].cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if preds[i].item() == labels[i].item() else 'red'
        plt.title(f"P: {preds[i].item()}\n(T: {labels[i].item()})", color=color)
        plt.axis('off')
    plt.suptitle("Debug: Test Set Predictions (P=Predicted, T=True)")
    plt.show()


# 4. 绘图函数：训练曲线
def plot_metrics(losses, accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, '-o', label='Training Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, '-s', color='orange', label='Test Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=64, custom_transform=my_transform)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    # 开始实验
    for epoch in range(1, 6):  # 训练5轮观察曲线更明显
        loss = train_model(model, device, train_loader, optimizer, epoch)

        # 这里的 test_model 逻辑复用之前的，只需要返回准确率
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch} 完成! 准确率: {acc:.2f}%")

        train_losses.append(loss)
        test_accuracies.append(acc)

    # 结果直观评估
    print("\n[结果评估] 绘制训练曲线与预测采样...")
    plot_metrics(train_losses, test_accuracies)
    visualize_predictions(model, device, test_loader)