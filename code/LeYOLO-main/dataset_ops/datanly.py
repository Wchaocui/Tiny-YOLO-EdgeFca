import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

'''统计结果的亮度/噪声'''
def calculate_brightness(img):
    """计算图像区域的平均亮度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def calculate_noise(img):
    """计算图像区域的噪声水平（使用标准差法）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    return np.std(noise)


# 配置路径
image_dir = "path/to/images"  # 图像文件夹路径
label_dir = "path/to/labels"  # YOLO标签文件夹路径
output_dir = "results"  # 结果保存路径
os.makedirs(output_dir, exist_ok=True)

# 初始化存储结构
conf_groups = {
    '0.0-0.2': {'brightness': [], 'noise': []},
    '0.2-0.4': {'brightness': [], 'noise': []},
    '0.4-0.6': {'brightness': [], 'noise': []},
    '0.6-0.8': {'brightness': [], 'noise': []},
    '0.8-1.0': {'brightness': [], 'noise': []}
}

# 处理所有图像
for img_file in tqdm(os.listdir(image_dir)):
    if not img_file.endswith(('.jpg', '.png', '.jpeg')):
        continue

    # 读取图像
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # 读取对应标签
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:  # 跳过无效行
            continue

        # 解析YOLO格式 (class, conf, x_center, y_center, width, height)
        conf = float(parts[1])
        x_center = float(parts[2]) * w
        y_center = float(parts[3]) * h
        box_w = float(parts[4]) * w
        box_h = float(parts[5]) * h

        # 计算边界框坐标
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 裁剪目标区域
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:  # 跳过空区域
            continue

        # 计算亮度和噪声
        brightness = calculate_brightness(roi)
        noise = calculate_noise(roi)

        # 按置信度分组存储
        if conf < 0.2:
            group = '0.0-0.2'
        elif conf < 0.4:
            group = '0.2-0.4'
        elif conf < 0.6:
            group = '0.4-0.6'
        elif conf < 0.8:
            group = '0.6-0.8'
        else:
            group = '0.8-1.0'

        conf_groups[group]['brightness'].append(brightness)
        conf_groups[group]['noise'].append(noise)

# 可视化并保存结果
plt.figure(figsize=(15, 10))

# 亮度分布
plt.subplot(2, 1, 1)
for group, data in conf_groups.items():
    plt.hist(data['brightness'], bins=50, alpha=0.5, label=group)
plt.title('Brightness Distribution by Confidence Group')
plt.xlabel('Brightness Value (0-255)')
plt.ylabel('Frequency')
plt.legend()

# 噪声分布
plt.subplot(2, 1, 2)
for group, data in conf_groups.items():
    plt.hist(data['noise'], bins=50, alpha=0.5, label=group)
plt.title('Noise Distribution by Confidence Group')
plt.xlabel('Noise Level (Standard Deviation)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'brightness_noise_distribution.png'))
plt.close()

# 保存统计数据
with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
    for group, data in conf_groups.items():
        f.write(f"Confidence Group: {group}\n")
        f.write(f"  Brightness - Mean: {np.mean(data['brightness']):.2f}, Std: {np.std(data['brightness']):.2f}\n")
        f.write(f"  Noise - Mean: {np.mean(data['noise']):.2f}, Std: {np.std(data['noise']):.2f}\n")
        f.write("-" * 50 + "\n")

print("Analysis completed! Results saved to:", output_dir)