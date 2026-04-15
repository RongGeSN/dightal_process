# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont


def imwrite_ch(path, img):
    """底层函数：支持中文路径的文件保存"""
    res, img_encode = cv2.imencode('.jpg', img)
    if res: img_encode.tofile(path)


def batch_gen_plates(count=20, save_dir="./test_dataset"):
    """批量合成模拟车牌图像"""
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 80)

    provinces = "京沪津渝冀豫云辽黑湘鲁赣粤晋苏浙鄂"
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    print(f"--- 步骤 2: 正在生成 {count} 张模拟测试车牌 ---")
    for i in range(count):
        # 随机生成车牌号：省份 + 城市代码 + 5位随机码
        text = random.choice(provinces) + random.choice(letters) + "".join(random.sample(letters + "0123456789", 5))

        # 绘制蓝色车牌背景
        plate = np.zeros((140, 440, 3), dtype=np.uint8)
        plate[:, :] = [255, 0, 0]

        pil_img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil_img).text((20, 20), text, font=font, fill=(255, 255, 255))
        plate = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 随机添加模糊效果，模拟相机抖动
        if random.random() > 0.5:
            plate = cv2.GaussianBlur(plate, (5, 5), 0)

        # 保存：文件名采用 ID_车牌号 的格式，非常利于管理和升级
        file_path = os.path.join(save_dir, f"{i:03d}_{text}.jpg")
        imwrite_ch(file_path, plate)
    print(f"测试集已保存至: {save_dir}\n")


if __name__ == "__main__":
    batch_gen_plates(40)