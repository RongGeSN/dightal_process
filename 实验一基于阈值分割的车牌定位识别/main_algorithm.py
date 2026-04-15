# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


class LPR_Deep_Tutorial:
    def __init__(self, template_dir="./templates"):
        self.template_lib = self._build_library(template_dir)

    def _build_library(self, path):
        """步骤3：建库"""
        lib = {}
        if not os.path.exists(path): return lib
        for f in os.listdir(path):
            char_name = f.split(".")[0]
            img = cv2.imdecode(np.fromfile(os.path.join(path, f), dtype=np.uint8), 0)
            if img is not None:
                lib[char_name] = cv2.resize(img, (20, 40))
        return lib

    def run_canny_steps(self, gray_img):
        """
        要求1：Canny全流程演示
        """
        # --- [参数调点 1] ---
        # (5, 5) 是滤波器大小，越大图像越模糊，噪点越少，但边缘也会变弱
        # 2. 高斯滤波
        # 调参建议：
        # - 默认用 (5, 5) 是比较稳妥的平衡点
        # - 如果环境光线暗、噪点多，尝试 (7, 7)
        # - 如果车牌本身很小、很清晰，尝试 (3, 3)
        step1_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        cv2.imshow("Canny_1_Preprocessing(Blur)", step1_blur)

        # 1.2 显式梯度计算 (使用 Sobel 算子)
        # 计算 X 方向梯度 (检测垂直边缘)
        grad_x = cv2.Sobel(step1_blur, cv2.CV_16S, 1, 0, ksize=3)

        # 计算 Y 方向梯度 (检测水平边缘)
        grad_y = cv2.Sobel(step1_blur, cv2.CV_16S, 0, 1, ksize=3)

        print(grad_x, grad_y)
        # 转换回 uint8 格式显示
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)

        # 合并梯度 (梯度处理)
        grad_total = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        cv2.imshow("2_Gradient_Calculation_Total", grad_total)

        # --- [参数调点 2] ---
        # 100 是低阈值，200 是高阈值。
        # 如果边缘断断续续，请调低高阈值；如果杂乱边缘太多，请调高低阈值。
        edges = cv2.Canny(step1_blur, 100, 200)
        cv2.imshow("Canny_2_3_4_Gradient_Extraction", edges)

        return edges

    # ================= [新增：分割字符逻辑] =================
    def segment_chars(self, plate_bin):
        """利用投影法把粘连的块切成单个字"""
        # 1. 确保输入的是二维矩阵 (Height, Width)
        if len(plate_bin.shape) > 2:
            plate_bin = cv2.cvtColor(plate_bin, cv2.COLOR_BGR2GRAY)

        # 2. 垂直投影：统计每一列有多少白色像素
        # 注意：np.sum(..., axis=0) 会得到一个长度等于图像宽度的数组
        vertical_proj = np.sum(plate_bin, axis=0) / 255

        # --- 修复点：确保它是一个数组 ---
        if not isinstance(vertical_proj, np.ndarray):
            vertical_proj = np.array([vertical_proj])

        char_bounds = []
        in_char = False
        start = 0
        threshold = 5  # 列像素阈值

        # 3. 遍历每一列的像素和
        for i in range(len(vertical_proj)):  # 使用 range(len()) 比直接 enumerate 更稳健
            val = vertical_proj[i]
            if val > threshold and not in_char:
                start = i
                in_char = True
            elif val <= threshold and in_char:
                char_bounds.append((start, i))
                in_char = False

        # 4. 提取单个字符并归一化
        char_imgs = []
        for (s, e) in char_bounds:
            # 过滤掉宽度太小的噪点 (车牌字符通常有一定宽度)
            if e - s > 10:
                char = plate_bin[:, s:e]
                # 统一尺寸为 20x40
                char_imgs.append(cv2.resize(char, (20, 40)))

        return char_imgs

    # ================= [新增：模板匹配逻辑] =================
    def match_chars(self, char_imgs):
        """核心比对逻辑：挨个‘找茬’"""
        result = ""
        for i, char_img in enumerate(char_imgs):
            best_score = -1
            best_char = "?"
            for name, template in self.template_lib.items():
                # 使用相关系数匹配，1表示完全一样
                res = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                if score > best_score:
                    best_score = score
                    best_char = name
            result += best_char
            # 顺便显示一下切下来的字
            cv2.imshow(f"Char_{i}", char_img)
        return result
    def process_and_learn(self, img_path):
        src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is None: return
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 展示 Canny 步骤
        edges = self.run_canny_steps(gray)

        # 形态学定位逻辑
        # --- [参数调点 3] ---
        # (17, 5) 矩形核的大小。如果车牌字符连不起来，调大这两个数。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Morphology_Closing", closed)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # --- [参数调点 4] ---
            # 2.2 和 5.5 是车牌的长宽比范围，如果你的车牌没被框住，检查这个比例。
            if 2.2 < w / h < 5.5 and w > 80:
                # 必须先抠出车牌 ROI，再进行二值化，最后才能投影
                plate_roi = src[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                # 要求2：阈值分割
                # OTSU 算法会自动帮你确定阈值
                plate_bin, thresh_val = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                print(f"自动确定的分割阈值为: {thresh_val}")
                cv2.imshow("Step2_Threshold_Segmentation", plate_bin)

                # 3. 执行识别逻辑 [新增调用]
                char_imgs = self.segment_chars(plate_bin)
                plate_num = self.match_chars(char_imgs)
                print(f"--- 识别结果: {plate_num} ---")

                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        images = sorted([f for f in os.listdir(test_path) if f.endswith('.jpg')])
        for name in images:
            print(f"检测到第{name}张车牌")
            lpr.process_and_learn(os.path.join(test_path, name))