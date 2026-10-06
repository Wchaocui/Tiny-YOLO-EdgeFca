import os
import shutil
from pathlib import Path

# ==================== 请修改以下路径 ====================

# 图片文件夹路径（根据图片名称来检索）
IMAGES_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/images/train"

# 源标签文件夹路径（标签文件当前所在位置）
SOURCE_LABELS_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/labels/train"  # ← 请修改为你的标签源目录

# 目标输出目录（你指定的路径）
OUTPUT_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/labels/train"

# 标签文件后缀（通常是 .txt，如果是其他格式请修改）
LABEL_EXT = ".txt"

# True = 复制（保留源文件），False = 移动（删除源文件）
COPY_MODE = True


# ====================================================

def match_and_transfer_labels(images_dir, source_labels_dir, output_dir,
                              label_ext='.txt', copy=True):
    """
    根据图片文件名匹配，将对应标签复制/移动到目标目录
    """
    images_dir = Path(images_dir)
    source_labels_dir = Path(source_labels_dir)
    output_dir = Path(output_dir)

    # 创建输出目录（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件名（不含后缀）
    image_stems = set()
    valid_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    for img_path in images_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_img_exts:
            image_stems.add(img_path.stem)

    print(f"📷 找到 {len(image_stems)} 张图片")
    print(f"📁 图片目录: {images_dir}")
    print(f"📁 源标签目录: {source_labels_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 模式: {'复制' if copy else '移动'}")
    print("-" * 50)

    matched = 0
    missing = []

    for stem in sorted(image_stems):
        label_file = source_labels_dir / f"{stem}{label_ext}"

        # 尝试精确匹配
        if not label_file.exists():
            # 尝试大小写不敏感匹配（遍历目录查找）
            for f in source_labels_dir.glob(f"*{label_ext}"):
                if f.stem.lower() == stem.lower():
                    label_file = f
                    break

        if label_file.exists():
            dest = output_dir / f"{stem}{label_ext}"
            if copy:
                shutil.copy2(label_file, dest)
            else:
                shutil.move(str(label_file), str(dest))
            matched += 1
        else:
            missing.append(stem)

    # 统计结果
    print(f"\n✅ 成功匹配: {matched}/{len(image_stems)}")

    if missing:
        print(f"\n⚠️  缺少标签 ({len(missing)}个):")
        for m in missing[:15]:
            print(f"   - {m}")
        if len(missing) > 15:
            print(f"   ... 还有 {len(missing) - 15} 个")

    # 检查输出目录中的文件数
    output_files = list(output_dir.glob(f"*{label_ext}"))
    print(f"\n📂 输出目录现有标签文件: {len(output_files)}个")

    return matched, missing


# 执行
if __name__ == "__main__":
    matched, missing = match_and_transfer_labels(
        images_dir=IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        label_ext=LABEL_EXT,
        copy=COPY_MODE
    )