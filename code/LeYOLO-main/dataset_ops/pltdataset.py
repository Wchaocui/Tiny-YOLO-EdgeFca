import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# ===================== Configuration (MODIFY HERE) =====================
LABELS_ROOT = r"/home/s0433/Downloads/data/spark-2022-stream-1/SLLFS/labels"
IMAGES_ROOT = r"/home/s0433/Downloads/data/spark-2022-stream-1/SLLFS/images"
CLASS_NAMES = ["smart_1", "cheops", "lisa_path", "debris", "proba3_ocs",
               "proba3_csc", "soho", "earth_ob1", "proba_2", "x_newton", "d_star"]
# ======================================================================

SUBSETS = ["train", "val", "test"]

all_category_counts = defaultdict(int)
all_brightness = []
all_bbox_sizes = []
subset_data = {s: {"category_counts": defaultdict(int), "brightness": [], "bbox_sizes": []} for s in SUBSETS}

def process_dataset(subset):
    label_dir = os.path.join(LABELS_ROOT, subset)
    img_dir = os.path.join(IMAGES_ROOT, subset)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        img_name = os.path.splitext(label_file)[0]
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            temp_path = os.path.join(img_dir, img_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if not img_path:
            print(f"Warning: Image not found -> {img_name}")
            continue

        # Read labels
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cls_id, x, y, w, h = line.split()
            cls_id = int(cls_id)
            w, h = float(w), float(h)

            subset_data[subset]["category_counts"][cls_id] += 1
            all_category_counts[cls_id] += 1

            # 目标占整张图的面积比（正确）
            bbox_size = w * h
            subset_data[subset]["bbox_sizes"].append(bbox_size)
            all_bbox_sizes.append(bbox_size)

        # ===================== 核心修改：灰度图 + 中位数亮度 =====================
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算 中位数 作为亮度（你要的）
        brightness = np.median(gray)
        # 保存
        subset_data[subset]["brightness"].append(brightness)
        all_brightness.append(brightness)

# Process all subsets
for s in SUBSETS:
    print(f"Processing {s}...")
    process_dataset(s)

# ===================== Plot Style =====================
plt.style.use('default')
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff9896"]

# =============================================================================
# 1. Category Count Plot
# =============================================================================
def plot_category_stats():
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Dataset Category Distribution", fontsize=16, weight="bold")

    classes = sorted(all_category_counts.keys())
    name_list = [CLASS_NAMES[c] for c in classes]

    def plot_single(ax, data, title):
        counts = [data.get(c, 0) for c in classes]
        sns.barplot(x=name_list, y=counts, ax=ax, palette=colors[:len(name_list)])
        ax.set_title(title, weight="bold")
        ax.set_ylabel("Number of Instances")
        ax.tick_params(axis='x', rotation=45)
        # 每个柱子标数量
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts)*0.01, str(v), ha="center", weight="bold", fontsize=9)

    plot_single(axes[0,0], all_category_counts, "Total Dataset")
    plot_single(axes[0,1], subset_data["train"]["category_counts"], "Train Set")
    plot_single(axes[1,0], subset_data["val"]["category_counts"], "Val Set")
    plot_single(axes[1,1], subset_data["test"]["category_counts"], "Test Set")

    plt.tight_layout()
    plt.savefig("category_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 2. 亮度：灰度图中位数分布，范围 0~40
# =============================================================================
def plot_brightness_stats():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Image Brightness Distribution (Grayscale Median)", fontsize=16, weight="bold")

    def plot_hist(ax, data, title):
        ax.hist(data, bins=25, color="#2D4263", alpha=0.9, edgecolor="white", linewidth=0.8)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Median Brightness (0 - 40)")
        ax.set_ylabel("Number of Images")
        ax.set_xlim(10, 50)  # 严格按你的数据范围
        ax.grid(alpha=0.3)

    plot_hist(axes[0,0], all_brightness, "Total Dataset")
    plot_hist(axes[0,1], subset_data["train"]["brightness"], "Train Set")
    plot_hist(axes[1,0], subset_data["val"]["brightness"], "Val Set")
    plot_hist(axes[1,1], subset_data["test"]["brightness"], "Test Set")

    plt.tight_layout()
    plt.savefig("brightness_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 3. BBox Size (Normalized Area Ratio)
# =============================================================================
def plot_bbox_size_stats():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Object Size Distribution (Normalized Area Ratio)", fontsize=16, weight="bold")

    def plot_hist(ax, data, title):
        ax.hist(data, bins=40, color="#C84B31", alpha=0.9, edgecolor="white", linewidth=0.5)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Object Area / Image Area")
        ax.set_ylabel("Number of Objects")
        ax.grid(alpha=0.3)

    plot_hist(axes[0,0], all_bbox_sizes, "Total Dataset")
    plot_hist(axes[0,1], subset_data["train"]["bbox_sizes"], "Train Set")
    plot_hist(axes[1,0], subset_data["val"]["bbox_sizes"], "Val Set")
    plot_hist(axes[1,1], subset_data["test"]["bbox_sizes"], "Test Set")

    plt.tight_layout()
    plt.savefig("bbox_size_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()

# ===================== Run =====================
if __name__ == "__main__":
    print("Generating visualizations...")
    plot_category_stats()
    plot_brightness_stats()
    plot_bbox_size_stats()
    print("All figures saved successfully!")
