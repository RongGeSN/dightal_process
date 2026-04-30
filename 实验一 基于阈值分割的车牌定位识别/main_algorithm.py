# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


class LPR_Deep_Tutorial:
    def __init__(self, template_dir="./templates"):
        # 移除了模板库的加载逻辑，减少环境依赖报错
        pass

    def run_canny_steps(self, gray_img):
        """
        要求1：Canny全流程演示
        """
        # 1. 高斯滤波：平滑图像，减少噪点
        step1_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        cv2.imshow("Canny_1_Preprocessing(Blur)", step1_blur)

        # 1.2 显式梯度计算 (使用 Sobel 算子)
        grad_x = cv2.Sobel(step1_blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(step1_blur, cv2.CV_16S, 0, 1, ksize=3)

        # 转换回 uint8 格式显示
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)

        # 合并梯度
        grad_total = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        cv2.imshow("2_Gradient_Calculation_Total", grad_total)

        # 2. Canny 边缘提取（包含非极大值抑制和双阈值滞后处理）
        edges = cv2.Canny(step1_blur, 100, 200)
        cv2.imshow("Canny_2_3_4_Gradient_Extraction", edges)

        return edges

    def process_and_learn(self, img_path):
        # 使用 np.fromfile 解决中文路径读取问题
        src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is None:
            print(f"无法读取图片: {img_path}")
            return

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 1. 运行边缘检测
        edges = self.run_canny_steps(gray)

        # 2. 形态学闭运算：将破碎的边缘连接成闭合区域（针对车牌字符密集区域）
        # (27, 5) 结构元，主要为了水平方向连接字符
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Morphology_Closing", closed)

        # 3. 轮廓查找
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # 4. 尺寸与长宽比过滤（中国车牌长宽比大约在 3:1 左右）
            aspect_ratio = w / float(h)
            if 2.2 < aspect_ratio < 5.5 and w > 80:
                # 提取车牌感兴趣区域 (ROI)
                plate_roi = src[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                # 5. 阈值分割演示 (OTSU 二值化)
                plate_bin, thresh_val = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print(f"检测到车牌区域，自动阈值: {thresh_val}")

                cv2.imshow("Step2_Threshold_Segmentation", plate_bin)

                # 在原图画出框
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(src, "License Plate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Final_Locate", src)
                break

        print("按任意键继续下一张...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lpr = LPR_Deep_Tutorial()
    # 自动遍历数据集
    test_path = "./test_dataset"
    if os.path.exists(test_path):
        images = sorted([f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if not images:
            print("目录下没有找到图片文件。")
        for name in images:
            full_path = os.path.join(test_path, name)
            print(f"正在处理: {name}")
            lpr.process_and_learn(full_path)
    else:
        print(f"路径不存在: {test_path}")