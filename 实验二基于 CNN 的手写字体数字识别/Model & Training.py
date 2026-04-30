# ==================== Model & Training.py ====================
import torch
import torch.nn as nn  # 神经网络模块，包含各种层和损失函数
import torch.optim as optim  # 优化器模块，用于更新模型参数
import matplotlib.pyplot as plt  # 绘图库，用于可视化训练过程
import random  # 随机模块，用于随机采样
import numpy as np  # 数值计算库

# 导入前两部分的模块
from Preprocessing import my_transform  # 预处理变换
from Data_Loader import get_mnist_loaders  # 数据加载器


# ==========================================
# 1. 搭建 CNN 网络 (SmartMnistCNN)
# ==========================================
class SmartMnistCNN(nn.Module):
    """
    智能MNIST分类卷积神经网络

    网络结构说明：
    - 输入：1x28x28的灰度图像
    - 卷积层：提取图像特征
    - 池化层：降低特征图尺寸，减少计算量
    - 全连接层：将特征映射到10个类别（数字0-9）

    改进特性：
    - BatchNorm：加速收敛，稳定训练
    - Dropout：防止过拟合
    """

    def __init__(self):
        """初始化网络层结构"""
        super(SmartMnistCNN, self).__init__()

        # ========== 特征提取部分：卷积层 ==========
        # 使用Sequential容器，按顺序执行各层操作
        self.features = nn.Sequential(
            # 第一层：卷积层
            # nn.Conv2d(输入通道数, 输出通道数, 卷积核大小, padding)
            # 输入：1通道灰度图，输出：32个特征图
            # padding=1：保持特征图尺寸不变（28x28 -> 28x28）
            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            # BatchNorm2d：批量归一化
            # 参数32表示对32个通道进行归一化
            # 作用：加速训练收敛，允许使用更大的学习率
            nn.BatchNorm2d(32),

            # ReLU激活函数：引入非线性
            # ReLU(x) = max(0,x)，负数变为0，正数保持不变
            nn.ReLU(),

            # 最大池化层：下采样
            # kernel_size=2：池化窗口大小2x2，步长默认等于窗口大小
            # 输出尺寸：28x28 -> 14x14（尺寸减半）
            nn.MaxPool2d(2),

            # 第二层：卷积层
            # 输入：32通道，输出：64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第二次池化：14x14 -> 7x7
            nn.MaxPool2d(2),

            # Dropout层：随机丢弃25%的神经元
            # 作用：防止过拟合，提高模型泛化能力
            nn.Dropout(0.25)
        )

        # ========== 分类部分：全连接层 ==========
        # 输入：卷积层输出的特征图展平后的向量
        # 64 * 7 * 7：64个通道，每个特征图7x7
        self.classifier = nn.Sequential(
            # Flatten：将多维张量展平为一维
            # 输入形状：(batch_size, 64, 7, 7) -> (batch_size, 64*7*7)
            nn.Flatten(),

            # 第一个全连接层
            # 输入：64*7*7=3136，输出：128
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # Dropout：丢弃50%的神经元（比卷积层更强的正则化）
            nn.Dropout(0.5),

            # 输出层
            # 输入：128，输出：10（10个数字类别）
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入图像张量，形状(batch_size, 1, 28, 28)

        返回:
            output (torch.Tensor): 输出logits，形状(batch_size, 10)
        """
        # 通过特征提取部分
        x = self.features(x)
        # 通过分类器部分
        output = self.classifier(x)
        return output


# ==========================================
# 2. 训练、评估与随机可视化函数
# ==========================================

def train_and_evaluate():
    """
    完整的训练和评估流程

    执行步骤：
    1. 加载数据
    2. 初始化模型、优化器、损失函数
    3. 训练多个epoch
    4. 绘制训练曲线
    5. 可视化预测结果
    """

    # ========== 设备配置 ==========
    # 检查是否有CUDA（NVIDIA GPU）可用
    # 如果有GPU则使用GPU加速，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # ========== A. 获取数据加载器 ==========
    # batch_size=64：每次训练64张图像
    # custom_transform=my_transform：应用形态学预处理
    train_loader, test_loader, _ = get_mnist_loaders(
        batch_size=64,
        custom_transform=my_transform
    )

    # ========== B. 初始化模型、优化器和损失函数 ==========
    # 创建模型实例并移动到指定设备
    model = SmartMnistCNN().to(device)

    # Adam优化器
    # 参数：model.parameters()表示优化所有模型参数
    # lr=0.001：学习率，控制参数更新的步长
    # Adam自适应调整学习率，相比SGD更容易调参
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 交叉熵损失函数
    # 适用于多分类问题，结合了LogSoftmax和NLLLoss
    # 输入：模型输出的logits，不需要额外做softmax
    criterion = nn.CrossEntropyLoss()

    # 记录训练过程中的损失和准确率，用于后续绘图
    history = {'train_loss': [], 'test_acc': []}

    # ========== C. 执行训练循环 ==========
    epochs = 3  # 训练3轮

    for epoch in range(1, epochs + 1):
        # 设置为训练模式
        # 作用：启用Dropout和BatchNorm的训练模式
        model.train()

        # 记录当前epoch的总损失
        running_loss = 0.0

        # 遍历训练数据
        # batch_idx：批次索引，data：图像张量，target：标签
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移动到指定设备（GPU或CPU）
            data, target = data.to(device), target.to(device)

            # 梯度清零
            # 因为梯度默认累加，每个batch前需要清零
            optimizer.zero_grad()

            # 前向传播：输入图像，得到预测结果
            output = model(data)

            # 计算损失：比较预测值和真实标签
            loss = criterion(output, target)

            # 反向传播：计算梯度
            loss.backward()

            # 更新参数：根据梯度调整模型权重
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 每200个batch打印一次训练信息
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # ========== 每个Epoch结束后评估测试集 ==========
        # 设置为评估模式
        # 作用：禁用Dropout，BatchNorm使用全局统计量
        model.eval()

        correct = 0  # 正确预测的样本数
        total = 0  # 总样本数

        # 不计算梯度，减少内存占用和计算量
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # 前向传播
                preds = model(data)

                # 获取预测类别（最大值对应的索引）
                # argmax(dim=1)：在第1维度（类别维度）上找最大值索引
                predictions = preds.argmax(dim=1)

                # 统计正确数
                # eq：逐元素比较，返回布尔张量，sum()计算True的数量
                correct += predictions.eq(target).sum().item()

        # 计算准确率百分比
        acc = 100. * correct / len(test_loader.dataset)

        # 记录历史数据
        history['train_loss'].append(running_loss / len(train_loader))  # 平均损失
        history['test_acc'].append(acc)

        print(f"==> Epoch {epoch} 结束，测试集准确率: {acc:.2f}%")

    # ========== D. 绘制训练指标曲线 ==========
    plot_training_curves(history)

    # ========== E. 随机抽取测试样本进行可视化 ==========
    visualize_random_predictions(model, device, test_loader)


def plot_training_curves(history):
    """
    绘制训练损失曲线和测试准确率曲线

    参数:
        history (dict): 包含'train_loss'和'test_acc'的字典
    """

    # 创建图形，大小为12x5英寸
    plt.figure(figsize=(12, 5))

    # 左子图：训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], '-o', label='Train Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")  # x轴标签
    plt.ylabel("Loss")  # y轴标签
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例

    # 右子图：测试准确率曲线
    plt.subplot(1, 2, 2)
    # '-s'：实线加方块标记，color='orange'：橙色线条
    plt.plot(history['test_acc'], '-s', color='orange', label='Test Accuracy')
    plt.title("Test Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    # 自动调整子图间距，避免重叠
    plt.tight_layout()

    # 显示图形
    plt.show()


def visualize_random_predictions(model, device, test_loader, num_samples=10):
    """
    随机抽取测试样本并可视化预测结果

    作用：直观展示模型的识别效果，用颜色区分正确/错误预测

    参数:
        model: 训练好的模型
        device: 设备（CPU/GPU）
        test_loader: 测试数据加载器
        num_samples (int): 要显示的样本数量，默认10
    """

    # 设置为评估模式
    model.eval()

    # 获取测试集的完整数据集对象
    dataset = test_loader.dataset
    total_samples = len(dataset)

    # 随机产生不重复的索引
    # random.sample：从序列中随机选择指定数量的不重复元素
    indices = random.sample(range(total_samples), num_samples)

    # 创建图形
    plt.figure(figsize=(18, 4))

    # 在画布左侧添加图例说明
    info_text = "Legend:\nGreen: Correct\nRed: Incorrect\nT: True Label\nP: Predicted"
    plt.gcf().text(0.01, 0.5, info_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.5))

    # 禁用梯度计算（推理模式）
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取原始图像和标签
            img_tensor, label = dataset[idx]

            # 准备输入数据：添加batch维度
            # unsqueeze(0)：在位置0添加一个维度，形状(1, 1, 28, 28)
            input_data = img_tensor.unsqueeze(0).to(device)

            # 模型推理
            output = model(input_data)
            pred = output.argmax(dim=1).item()  # 预测的类别

            # 创建子图
            ax = plt.subplot(1, num_samples, i + 1)

            # 将张量转换为可显示的图像
            # squeeze()：去除batch维度，numpy()：转numpy数组
            img_show = img_tensor.squeeze().numpy()
            plt.imshow(img_show, cmap='gray')  # 灰度图显示

            # 根据预测是否正确设置标题颜色
            # 正确：绿色，错误：红色
            color = 'green' if pred == label else 'red'

            # 设置标题：显示真实标签(T)和预测标签(P)
            ax.set_title(f"T: {label}\nP: {pred}",
                         color=color, fontsize=14, fontweight='bold')

            # 关闭坐标轴
            plt.axis('off')

    # 设置总标题
    plt.suptitle("CNN Recognition Results (Random Samples from Test Set)",
                 fontsize=16, y=1.08)

    # 显示图形
    plt.show()


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 只有直接运行此文件时才执行训练
    # 如果被其他文件导入，则不自动执行
    train_and_evaluate()