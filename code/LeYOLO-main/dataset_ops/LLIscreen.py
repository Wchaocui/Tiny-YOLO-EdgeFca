import cv2
import numpy as np

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
def get_target_features(img_path, yolo_boxes, dark_thresh=20, kernel_size=3):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, kernel_size)
    h_img, w_img = gray.shape[:2]
    obj_pixels = []
    bg_mask = np.ones((h_img, w_img), dtype=bool)

    for box in yolo_boxes:
        x, y, w, h = box
        x1 = int((x - w/2) * w_img)
        y1 = int((y - h/2) * h_img)
        x2 = int((x + w/2) * w_img)
        y2 = int((y + h/2) * h_img)

        obj_region = gray[y1:y2, x1:x2]
        valid = obj_region[obj_region > dark_thresh].flatten()
        obj_pixels.extend(valid)
        bg_mask[y1:y2, x1:x2] = False

    obj_pixels = np.array(obj_pixels) if len(obj_pixels) > 0 else np.array([0])
    target_median = round(np.median(obj_pixels), 2)
    target_brightness = round(np.mean(obj_pixels), 2)

    bg_pixels = gray[bg_mask & (gray > dark_thresh)].flatten()
    bg_brightness = round(np.mean(bg_pixels), 2) if len(bg_pixels) > 0 else 0
    contrast = round(target_brightness - bg_brightness, 2)

    return target_median, target_brightness, contrast

# --------------------------
# ✅ 低光判定函数（固定科学阈值，空间图像专用）
# --------------------------
def is_low_light(global_med, target_med, target_brightness,contrast):
    # 阈值专为深空图像设定
    # 全局中位数 ≤ 40  AND  目标亮度 ≤ 50 → 判定低光
    low_global = (global_med <= 40)
    low_target = (target_brightness <= 40)
    low_target_med = (target_med <= 40)
    low_contrast = (contrast<= 1)
    return low_global and low_target and low_target_med and low_contrast

# ==========================
# 测试 2 张图
# ==========================
if __name__ == "__main__":
    img1_path = "/home/s0433/local/projects_py/LeYOLO-main/picture/img000126.jpg"
    img2_path = "/home/s0433/local/projects_py/LeYOLO-main/picture/img001022.jpg"

    boxes1 = [[0.43017578125, 0.267578125, 0.8291015625, 0.412109375]]
    boxes2 = [[0.390625, 0.2724609375, 0.56640625, 0.544921875]]

    # ========== 图像1 ==========
    g_med1 = get_global_median(img1_path)
    t_med1, t_bright1, contrast1 = get_target_features(img1_path, boxes1)
    low1 = is_low_light(g_med1, t_med1, t_bright1,contrast1)

    print("===== 图像 1 =====")
    print(f"全局灰度中位数: {g_med1}")
    print(f"目标区域灰度中位数: {t_med1}")
    print(f"目标区域亮度: {t_bright1}")
    print(f"目标区域对比度: {contrast1}")
    print(f"是否低光照图像: {low1}\n")

    # ========== 图像2 ==========
    g_med2 = get_global_median(img2_path)
    t_med2, t_bright2, contrast2 = get_target_features(img2_path, boxes2)
    low2 = is_low_light(g_med2, t_med2, t_bright2,contrast2)

    print("===== 图像 2 =====")
    print(f"全局灰度中位数: {g_med2}")
    print(f"目标区域灰度中位数: {t_med2}")
    print(f"目标区域亮度: {t_bright2}")
    print(f"目标区域对比度: {contrast2}")
    print(f"是否低光照图像: {low2}")
