# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


class LPR_Deep_Tutorial:
    def __init__(self, template_dir="./templates"):
        # 步骤3：建库 - 载入字典
        self.template_lib = self._build_library(template_dir)

    def _build_library(self, path):
        """步骤3：建库"""
        lib = {}
        if not os.path.exists(path): return lib
        for f in os.listdir(path):
            char_name = f.split(".")[0]
            # 解决中文路径读取问题
            img = cv2.imdecode(np.fromfile(os.path.join(path, f), dtype=np.uint8), 0)
            if img is not None:
                # 统一尺寸非常重要，必须和匹配时一致
                lib[char_name] = cv2.resize(img, (20, 40))
        return lib

    def run_canny_steps(self, gray_img):
        """要求1：Canny全流程演示"""
        step1_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        cv2.imshow("Canny_1_Preprocessing(Blur)", step1_blur)

        grad_x = cv2.Sobel(step1_blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(step1_blur, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        grad_total = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        cv2.imshow("2_Gradient_Calculation_Total", grad_total)

        edges = cv2.Canny(step1_blur, 100, 200)
        cv2.imshow("Canny_2_3_4_Gradient_Extraction", edges)
        return edges

    def segment_chars(self, plate_bin):
        """利用投影法把粘连的块切成单个字"""
        if not hasattr(plate_bin, 'shape'): return []

        # 垂直投影
        vertical_proj = np.sum(plate_bin, axis=0) / 255

        char_bounds = []
        in_char = False
        start = 0
        threshold = 5

        for i in range(len(vertical_proj)):
            val = vertical_proj[i]
            if val > threshold and not in_char:
                start = i
                in_char = True
            elif val <= threshold and in_char:
                char_bounds.append((start, i))
                in_char = False

        char_imgs = []
        for (s, e) in char_bounds:
            # 这里的宽度过滤逻辑会影响识别准确率
            if 10 < (e - s) < 50:
                char = plate_bin[:, s:e]
                # 统一缩放到模板大小
                char_imgs.append(cv2.resize(char, (20, 40)))

        return char_imgs

    def match_chars(self, char_imgs):
        """核心比对逻辑：挨个‘找茬’"""
        result = ""
        for i, char_img in enumerate(char_imgs):
            best_score = -1
            best_char = "?"

            for name, template in self.template_lib.items():
                # 使用相关系数匹配
                res = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                if score > best_score:
                    best_score = score
                    best_char = name

            # 如果最高分都很低，说明匹配不可靠
            if best_score < 0.5:
                result += "？"
            else:
                result += best_char

            cv2.imshow(f"Char_Debug_{i}", char_img)
        return result

    def process_and_learn(self, img_path):
        src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is None: return
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edges = self.run_canny_steps(gray)

        # 定位
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Morphology_Closing", closed)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 2.2 < w / h < 5.5 and w > 80:
                plate_roi = src[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

                # 步骤2：阈值分割
                # 修复返回值顺序问题
                thresh_val, plate_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                cv2.imshow("Step2_Binary_Plate", plate_bin)

                # 执行识别
                char_imgs = self.segment_chars(plate_bin)
                plate_num = self.match_chars(char_imgs)
                print(f"--- 识别结果: {plate_num} ---")

                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Final_Locate", src)
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lpr = LPR_Deep_Tutorial()
    test_path = "./test_dataset"
    if os.path.exists(test_path):
        images = sorted([f for f in os.listdir(test_path) if f.endswith('.jpg')])
        for name in images:
            print(f"正在处理: {name}")
            lpr.process_and_learn(os.path.join(test_path, name))