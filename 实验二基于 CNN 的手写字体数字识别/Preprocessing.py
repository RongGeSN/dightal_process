import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


class MorphologicalTransform:
    """自定义形态学处理 + 归一化类"""

    def __call__(self, img):
        # PIL 转 Numpy
        img_np = np.array(img)

        # 形态学处理：开运算 (先腐蚀后膨胀，去除噪点)
        kernel = np.ones((2, 2), np.uint8)
        img_morph = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 转回 Tensor 并归一化
        final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return final_transform(img_morph)


def show_preprocessing_comparison():
    """Debug专用：展示形态学处理前后的效果"""
    raw_mnist = datasets.MNIST(root='./data', train=True, download=True)
    img_idx = 0  # 观察第1张图
    raw_img, label = raw_mnist[img_idx]

    # 手动执行一次处理逻辑
    processor = MorphologicalTransform()
    processed_tensor = processor(raw_img)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original (Label: {label})")
    plt.imshow(raw_img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("After Morphology (2x2 Open)")
    # 反向转换 Tensor 回图片用于显示
    plt.imshow(processed_tensor.squeeze().numpy(), cmap='gray')

    plt.show()


# 实例化对象供外部调用
my_transform = MorphologicalTransform()

if __name__ == "__main__":
    # 单独运行此文件可以查看预处理效果
    show_preprocessing_comparison()