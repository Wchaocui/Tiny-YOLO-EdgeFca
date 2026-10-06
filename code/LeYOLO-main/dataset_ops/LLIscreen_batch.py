import cv2
import numpy as np
import os
from pathlib import Path

# --------------------------
# 空间图像预处理：中值滤波 + 去微小噪声
# --------------------------
def preprocess_space_image(gray_img, dark_thresh=20, min_area=8, kernel_size=3):
    gray_blur = cv2.medianBlur(gray_img, kernel_size)
    _, binary = cv2.threshold(gray_blur, dark_thresh, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(gray_blur)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255
    gray_clean = gray_blur.copy()
    gray_clean[mask == 0] = 0
    return gray_clean

# --------------------------
# 全局灰度中位数
# --------------------------
def get_global_median(img_path, kernel_size=3):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clean = preprocess_space_image(gray, kernel_size=kernel_size)
    return round(np.median(gray_clean), 2)

# --------------------------
# 目标区域：中位数、亮度、对比度
# --------------------------
# 目标区域亮度统计
# --------------------------
def get_target_features(img_path, yolo_boxes, dark_thresh=20, kernel_size=3):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, kernel_size)
    h_img, w_img = gray.shape[:2]
    obj_pixels = []
    bg_mask = np.ones((h_img, w_img), dtype=bool)

    for box in yolo_boxes:
        if len(box) != 4:
            continue
        x, y, w, h = box
        x1 = int((x - w/2) * w_img)
        y1 = int((y - h/2) * h_img)
        x2 = int((x + w/2) * w_img)
        y2 = int((y + h/2) * h_img)

        obj_region = gray[y1:y2, x1:x2]
        valid = obj_region[obj_region > dark_thresh].flatten()
        obj_pixels.extend(valid)
        bg_mask[y1:y2, x1:x2] = False

    if len(obj_pixels) == 0:
        return 0.0, 0.0, 0.0

    obj_pixels = np.array(obj_pixels)
    target_median = round(np.median(obj_pixels), 2)
    target_brightness = round(np.mean(obj_pixels), 2)

    bg_pixels = gray[bg_mask & (gray > dark_thresh)].flatten()
    bg_brightness = round(np.mean(bg_pixels), 2) if len(bg_pixels) > 0 else 0
    contrast = round(target_brightness - bg_brightness, 2)

    return target_median, target_brightness, contrast
# --------------------------
# 低光判定函数（你自己的规则）
# --------------------------
def is_low_light(global_med, target_med, target_brightness, contrast):
    low_global = (global_med <= 40)
    low_target = (target_brightness <= 40)
    low_target_med = (target_med <= 40)
    low_contrast = (contrast <= 1)
    return low_global and low_target and low_target_med and low_contrast

# --------------------------
# 读取 YOLO 标注 txt 文件
# --------------------------
def load_yolo_boxes(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        if len(parts) >= 4:
            x, y, w, h = parts[1], parts[2], parts[3], parts[4]
            boxes.append([x, y, w, h])
    return boxes

# --------------------------
# 批量处理函数
# --------------------------
def batch_process_images(img_dir, label_dir=None, save_low_light_dir="/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/images/val"):
    Path(save_low_light_dir).mkdir(exist_ok=True)
    img_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
    low_count = 0
    total_count = 0

    print("=" * 60)
    print("开始批量处理低光图像筛选")
    print("=" * 60)

    for img_name in os.listdir(img_dir):
        sf = os.path.splitext(img_name)[-1].lower()
        if sf not in img_suffix:
            continue

        img_path = os.path.join(img_dir, img_name)
        total_count += 1

        # 读取标注
        if label_dir is not None:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(label_dir, txt_name)
            boxes = load_yolo_boxes(txt_path)
        else:
            boxes = []

        # 计算指标
        g_med = get_global_median(img_path)
        t_med, t_bright, contrast = get_target_features(img_path, boxes)
        low = is_low_light(g_med, t_med, t_bright, contrast)

        # 输出
        print(f"[{total_count}] {img_name}")
        print(f"  全局中位数: {g_med} | 目标中位数: {t_med} | 亮度: {t_bright} | 对比度: {contrast}")
        print(f"  是否低光: {low}\n")

        # 保存低光图
        if low:
            low_count += 1
            img = cv2.imread(img_path)
            save_path = os.path.join(save_low_light_dir, img_name)
            cv2.imwrite(save_path, img)

    print("=" * 60)
    print(f"处理完成！总计：{total_count} 张")
    print(f"筛选低光图像：{low_count} 张 | 已保存到：{save_low_light_dir}")
    print("=" * 60)

# ==========================
# 批量运行入口（只需改这里）
# ==========================
if __name__ == "__main__":
    IMG_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/images/train"  # 图片文件夹
    LABEL_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/labels/train" # 标注文件夹（和图片同目录就不改）

    batch_process_images(IMG_DIR, label_dir=LABEL_DIR,
                         save_low_light_dir="/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/images/train")
