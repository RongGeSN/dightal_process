# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


class LPR_Deep_Tutorial:
    def __init__(self, template_dir="./templates"):
        # 统一字符尺寸：这是模板匹配的关键，待测图和库图必须都是这个尺寸
        self.char_w = 20
        self.char_h = 40
        self.template_lib = self._build_library(template_dir)

    def _build_library(self, path):
        """步骤3：建库 - 识别图片要与库尺寸一致"""
        lib = {}
        if not os.path.exists(path):
            print(f"警告：模板库路径 {path} 不存在")
            return lib

        # 遍历文件夹，如 "A.jpg", "苏.jpg", "1.jpg"
        for f in os.listdir(path):
            if f.endswith(('.jpg', '.png', '.bmp')):
                char_name = f.split(".")[0]
                # 使用imdecode支持中文路径
                img = cv2.imdecode(np.fromfile(os.path.join(path, f), dtype=np.uint8), 0)
                if img is not None:
                    # 强制调整为标准尺寸 (20x40)
                    lib[char_name] = cv2.resize(img, (self.char_w, self.char_h))
        print(f"成功加载 {len(lib)} 个模板字符")
        return lib

    def run_canny_steps(self, gray_img):
        """Canny全流程"""
        step1_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges = cv2.Canny(step1_blur, 100, 200)
        return edges

    def segment_chars(self, plate_bin):
        """利用垂直投影法分割字符"""
        # 确保是二值图
        if len(plate_bin.shape) > 2:
            plate_bin = cv2.cvtColor(plate_bin, cv2.COLOR_BGR2GRAY)

        # 关键讲究：车牌识别通常需要“黑底白字”进行投影
        # 如果背景白字符黑，需翻转：plate_bin = cv2.bitwise_not(plate_bin)

        # 1. 垂直投影
        vertical_proj = np.sum(plate_bin, axis=0) / 255

        char_bounds = []
        in_char = False
        start = 0
        threshold = 5  # 噪声阈值，根据图片清晰度调整

        for i, val in enumerate(vertical_proj):
            if val > threshold and not in_char:
                start = i
                in_char = True
            elif val <= threshold and in_char:
                if i - start > 5:  # 过滤太窄的噪点
                    char_bounds.append((start, i))
                in_char = False

        char_imgs = []
        for (s, e) in char_bounds:
            char_roi = plate_bin[:, s:e]
            # 缩放到与模板库一致的尺寸
            resized_char = cv2.resize(char_roi, (self.char_w, self.char_h))
            char_imgs.append(resized_char)
        return char_imgs

    def match_chars(self, char_imgs):
        """核心比对逻辑：模板匹配"""
        if not self.template_lib:
            return "No Templates"

        result = ""
        for i, char_img in enumerate(char_imgs):
            best_score = -1
            best_char = "?"

            for name, template in self.template_lib.items():
                # TM_CCOEFF_NORMED：相关系数匹配，值越大越相似
                res = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(res)

                if score > best_score:
                    best_score = score
                    best_char = name

            result += best_char
            # 调试显示：查看切下来的字
            cv2.imshow(f"Seg_Char_{i}", char_img)
        return result

    def process_and_learn(self, img_path):
        src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is None: return
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 1. 定位逻辑
        edges = self.run_canny_steps(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 2.0 < w / h < 5.5 and w > 60:
                # 提取车牌 ROI
                plate_roi = gray[y:y + h, x:x + w]

                # 2. 阈值分割 (实验要求2)
                # 使用 OTSU 自动阈值，并根据需要决定是否要 cv2.THRESH_BINARY_INV
                _, plate_bin = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 重要：模板匹配通常识别“白字”，如果二值化后字是黑的，请反色
                # 假设车牌字符占比小于背景，我们可以通过像素统计判断是否反色
                if np.sum(plate_bin == 255) > np.sum(plate_bin == 0):
                    plate_bin = cv2.bitwise_not(plate_bin)

                cv2.imshow("Plate_Binary", plate_bin)

                # 3. 字符分割与识别 (实验要求3)
                char_list = self.segment_chars(plate_bin)
                plate_num = self.match_chars(char_list)

                print(f"车牌位置: [{x},{y}] 识别结果: {plate_num}")

                # 绘制结果
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(src, plate_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Result", src)
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 请确保 ./templates 文件夹下有字符图片
    lpr = LPR_Deep_Tutorial(template_dir="./templates")
    lpr.process_and_learn("./test_dataset/000_京QE9CQ4.jpg")  # 替换为你的测试图路径



'''
GitHub 地址： asishjose/license-plate-detection-opencv
GitHub 地址： MoazHassan2022/License-Plate-Recognition
话题：https://gemini.google.com/app/e966ea12ce221d8c
1. 经典入门型：License-Plate-Recognition (OpenCV + Template Matching)
这个项目几乎是为你这张实验单“量身定制”的。它完全避开了复杂的深度学习，只使用传统的图像处理方法。

核心流程： 灰度化 -> 边缘检测 (Canny) -> 轮廓寻找 -> 阈值分割 -> 模板匹配。

匹配点： 实验要求中的“字符归一化”和“建立库、比对”在代码中都有清晰的体现。

GitHub 地址： MoazHassan2022/License-Plate-Recognition
https://github.com/asishjose/license-plate-detection-opencv
特点： 逻辑简单，适合新手对照实验报告编写。

2. 流程最全型：Python_Plate_Recognition
这个项目在实现定位的同时，重点展示了图像预处理的每一个步骤（如高斯模糊、Sobel/Canny 边缘检测等）。

实验关联： 非常适合用来观察“梯度计算”和“边缘提取”的中间结果。

GitHub 地址： asishjose/license-plate-detection-opencv
https://github.com/MoazHassan2022/License-Plate-Recognition
特点： 代码注释详尽，方便你直接截取中间过程图作为实验报告的内容。
'''