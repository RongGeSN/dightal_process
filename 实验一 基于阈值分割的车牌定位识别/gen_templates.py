# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def generate_templates(save_path="./templates"):
    """生成字符识别所需的标准模板库"""
    if not os.path.exists(save_path): os.makedirs(save_path)

    # 定义需要识别的所有字符
    provinces = "京沪津渝冀豫云辽黑湘鲁赣粤晋苏浙鄂"
    alphabets = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "0123456789"
    chars = provinces + alphabets + digits

    # Windows系统常用字体路径，Linux用户需更换
    font_path = "C:/Windows/Fonts/simhei.ttf"
    font = ImageFont.truetype(font_path, 35)

    print("--- 步骤 1: 正在生成字符模板库 ---")
    for c in chars:
        # 创建 20x40 的黑底白字图像（这是OpenCV识别的标准尺寸）
        img_pil = Image.new('L', (20, 40), 0)
        draw = ImageDraw.Draw(img_pil)
        draw.text((2, 0), c, font=font, fill=255)

        img_cv = np.array(img_pil)
        save_name = os.path.join(save_path, f"{c}.png")

        # 使用编码技巧防止中文路径乱码
        cv2.imencode('.png', img_cv)[1].tofile(save_name)
    print(f"模板库已保存至: {save_path}\n")


if __name__ == "__main__":
    generate_templates()