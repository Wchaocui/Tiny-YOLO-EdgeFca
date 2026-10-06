import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ==================== 配置 ====================
# 原始数据路径（只读，不动）
SRC_ROOT = Path("/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected")

# 新数据路径（划分后的数据存放这里）
DST_ROOT = Path("/home/s0433/Downloads/data/spark-2022-stream-1/SLLFS")

SRC_IMAGES = SRC_ROOT / "images"
SRC_LABELS = SRC_ROOT / "labels"

DST_IMAGES = DST_ROOT / "images"
DST_LABELS = DST_ROOT / "labels"

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子
SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ==================== 1. 收集所有数据 ====================
random.seed(SEED)

all_samples = []

for split in ['train', 'val', 'test']:
    img_split_dir = SRC_IMAGES / split
    lbl_split_dir = SRC_LABELS / split

    if not img_split_dir.exists():
        print(f"警告: {img_split_dir} 不存在，跳过")
        continue

    # 收集存在的图像
    existing_images = {}
    for img_path in img_split_dir.iterdir():
        if img_path.suffix.lower() in IMG_EXTS:
            existing_images[img_path.stem] = img_path

    if not lbl_split_dir.exists():
        print(f"警告: {lbl_split_dir} 不存在，跳过")
        continue

    valid_count = 0
    missing_img = 0

    for lbl_path in sorted(lbl_split_dir.glob('*.txt')):
        stem = lbl_path.stem
        img_path = existing_images.get(stem)

        # 尝试其他扩展名
        if img_path is None:
            for ext in IMG_EXTS:
                candidate = img_split_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        if img_path is None or not img_path.exists():
            missing_img += 1
            continue
        if not lbl_path.exists():
            continue

        all_samples.append((img_path, lbl_path))
        valid_count += 1

    print(f"  {split}: 有效={valid_count}, 图像缺失={missing_img}")

print(f"\n总共收集到 {len(all_samples)} 个有效样本")

# ==================== 2. 随机打乱并划分 ====================
random.shuffle(all_samples)

n_total = len(all_samples)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)
n_test = n_total - n_train - n_val

train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:n_train + n_val]
test_samples = all_samples[n_train + n_val:]

print(f"\n划分结果:")
print(f"  训练集 (train): {len(train_samples)} ({len(train_samples) / n_total * 100:.1f}%)")
print(f"  验证集 (val):   {len(val_samples)} ({len(val_samples) / n_total * 100:.1f}%)")
print(f"  测试集 (test):  {len(test_samples)} ({len(test_samples) / n_total * 100:.1f}%)")


# ==================== 3. 创建新目录（不碰原始数据） ====================
def create_dir(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)


print(f"\n创建新目录: {DST_ROOT}")
for split in ['train', 'val', 'test']:
    create_dir(DST_IMAGES / split)
    create_dir(DST_LABELS / split)


# ==================== 4. 复制文件到新目录 ====================
def copy_samples(samples, split_name):
    success = 0
    failed = 0

    for img_path, lbl_path in samples:
        dst_img = DST_IMAGES / split_name / img_path.name
        dst_lbl = DST_LABELS / split_name / lbl_path.name

        try:
            if not img_path.exists() or not lbl_path.exists():
                failed += 1
                continue

            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            success += 1

        except Exception as e:
            print(f"  失败 {img_path.name}: {e}")
            failed += 1

    print(f"  {split_name}: 成功={success}, 失败={failed}")
    return success, failed


print(f"\n复制文件中...")
copy_samples(train_samples, 'train')
copy_samples(val_samples, 'val')
copy_samples(test_samples, 'test')

# ==================== 5. 验证 & 生成yaml ====================
print(f"\n{'=' * 50}")
print(f"验证结果:")
total_imgs = 0
total_lbls = 0
for split in ['train', 'val', 'test']:
    n_imgs = len([f for f in (DST_IMAGES / split).iterdir()
                  if f.suffix.lower() in IMG_EXTS])
    n_lbls = len(list((DST_LABELS / split).glob('*.txt')))
    total_imgs += n_imgs
    total_lbls += n_lbls
    print(f"  {split}: 图像={n_imgs}, 标签={n_lbls}, 匹配={'✅' if n_imgs == n_lbls else '❌'}")

print(f"\n总计: 图像={total_imgs}, 标签={total_lbls}")
print(f"新数据路径: {DST_ROOT}")

# ==================== 6. 自动生成新的yaml配置文件 ====================
yaml_content = f"""# 低光照航天器目标检测数据集 (重新划分)
# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {DST_ROOT}  # 数据集根目录
train: images/train
val: images/val
test: images/test

nc: 11  # 请修改为实际类别数
names:
  0: class0
  1: class1
  2: class2
  3: class3
  4: class4
  5: class5
  6: class6
  7: class7
  8: class8
  9: class9
  10: class10
"""

yaml_path = DST_ROOT / "SLLFS_resplit.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n配置文件已生成: {yaml_path}")
print(f"{'=' * 50}")