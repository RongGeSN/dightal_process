# ==================== Preprocessing.py ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


class MorphologicalTransform:
    """
    自定义图像预处理类
    作用：对MNIST手写数字图像进行形态学预处理，然后转换为张量并归一化
    形态学处理可以去除图像中的噪点，平滑数字边缘，提高识别准确率
    """

    def __call__(self, img):
        """
        将图像通过形态学处理和标准化。

        参数:
            img (PIL.Image): 输入的PIL格式图像

        返回:
            torch.Tensor: 处理后的图像张量，形状为(1, 28, 28)，值经过标准化
        """

        # 步骤1：将PIL图像转换为NumPy数组
        # PIL图像是RGB或灰度格式，转换为numpy数组后方便OpenCV处理
        img_np = np.array(img)

        # 步骤2：形态学处理 - 开运算（先腐蚀后膨胀）
        # 作用：去除图像中的小噪点，同时保持数字主体形状不变
        # 开运算 = 腐蚀 + 膨胀，可以有效去除孤立的黑点或白点

        # 创建结构元素（核），大小为2x2
        # kernel定义了形态学操作的范围，2x2表示每次检查2x2像素区域
        kernel = np.ones((2, 2), np.uint8)

        # 执行开运算：cv2.MORPH_OPEN表示开运算
        # img_morph是处理后的图像，噪点被去除
        img_morph = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 步骤3：转换为张量并归一化
        # transforms.Compose将多个变换组合成一个
        final_transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像或numpy数组转换为张量，值范围从[0,255]缩放到[0.0,1.0]
            transforms.Normalize((0.1307,), (0.3081,))  # 标准化：使用MNIST数据集的均值和标准差
            # 公式: output = (input - mean) / std
            # 使得数据分布接近均值为0，标准差为1，有助于模型收敛
        ])

        # 返回最终处理后的张量
        return final_transform(img_morph)


def show_preprocessing_comparison():
    """
    Debug专用函数：展示形态学处理前后的效果对比
    作用：可视化原始图像和处理后的图像，帮助理解预处理的效果
    """

    # 加载原始MNIST数据集（不带任何变换）
    raw_mnist = datasets.MNIST(root='./data', train=True, download=True)

    # 取第一张图像作为示例
    img_idx = 0  # 可以选择不同的索引查看不同图像
    raw_img, label = raw_mnist[img_idx]

    # 手动执行一次处理逻辑
    processor = MorphologicalTransform()
    processed_tensor = processor(raw_img)

    # 创建对比图
    plt.figure(figsize=(10, 5))

    # 左图：原始图像
    plt.subplot(1, 2, 1)
    plt.title(f"Original (Label: {label})")
    plt.imshow(raw_img, cmap='gray')
    plt.axis('off')  # 关闭坐标轴

    # 右图：处理后图像
    plt.subplot(1, 2, 2)
    plt.title("After Morphology (2x2 Open)")
    # 将张量转换回numpy数组用于显示，squeeze()去掉维度为1的维度
    plt.imshow(processed_tensor.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    # 显示图像
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# 实例化对象供外部调用
# 这样做的好处：其他模块可以直接导入my_transform使用，无需重新实例化
my_transform = MorphologicalTransform()

# 如果直接运行此文件，则执行预处理效果对比
if __name__ == "__main__":
    # 单独运行此文件可以查看预处理效果
    show_preprocessing_comparison()