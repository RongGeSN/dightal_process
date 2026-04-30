import cv2
import numpy as np
from torchvision import transforms


class MorphologyAndNormalize:
    """
    自定义预处理类：包含形态学处理和归一化
    """

    def __call__(self, img):
        # 1. 将 PIL 图像转换为 Numpy 格式供 OpenCV 处理
        img_np = np.array(img)

        # 2. 形态学处理：开运算 (去除细微噪声)
        # 建立 2x2 的卷积核
        kernel = np.ones((2, 2), np.uint8)
        img_morph = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 3. 统一尺寸：虽然 MNIST 已经是 28x28，但这里保留接口以满足实验要求
        img_resized = cv2.resize(img_morph, (28, 28))

        # 4. 转换回 Tensor 并归一化
        # 归一化参数为 MNIST 官方推荐值
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform_pipeline(img_resized)


# 实例化预处理对象
my_transform = MorphologyAndNormalize()