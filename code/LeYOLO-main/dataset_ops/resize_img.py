import os
from pathlib import Path
from PIL import Image
import shutil

# ==================== 配置路径 ====================

# 输入图片目录
INPUT_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/images/train1024"

# 输出裁剪后图片目录（会自动创建）
OUTPUT_DIR = "/home/s0433/Downloads/data/spark-2022-stream-1/low_light_selected/images/train"

# 目标尺寸
TARGET_SIZE = (640, 640)

# 裁剪模式: "center"=中心裁剪, "resize"=直接缩放, "pad"=等比缩放+填充黑边
MODE = "center"

# 图片格式转换（None保持原格式，或指定如".jpg", ".png"）
OUTPUT_EXT = None

# 是否覆盖已存在文件
OVERWRITE = False


# ====================================================

def crop_center(img, target_w, target_h):
    """中心裁剪"""
    w, h = img.size
    left = max(0, (w - target_w) // 2)
    top = max(0, (h - target_h) // 2)
    right = min(w, left + target_w)
    bottom = min(h, top + target_h)
    return img.crop((left, top, right, bottom))


def resize_with_pad(img, target_w, target_h, fill_color=(0, 0, 0)):
    """等比缩放并填充黑边，保持原图比例"""
    w, h = img.size
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # 创建背景并居中粘贴
    if img.mode == 'RGBA':
        background = Image.new('RGBA', (target_w, target_h), fill_color + (0,))
    elif img.mode == 'L':
        background = Image.new('L', (target_w, target_h), fill_color[0])
    else:
        background = Image.new('RGB', (target_w, target_h), fill_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    background.paste(resized, (paste_x, paste_y))
    return background


def batch_crop_images(input_dir, output_dir, target_size=(640, 640),
                      mode="center", output_ext=None, overwrite=False):
    """
    批量裁剪/缩放图片

    Args:
        mode: "center"中心裁剪, "resize"直接缩放, "pad"等比缩放+填充
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_w, target_h = target_size
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    processed = 0
    skipped = 0
    errors = []

    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in valid_exts]

    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 目标尺寸: {target_w}x{target_h}")
    print(f"🔧 裁剪模式: {mode}")
    print(f"📷 找到 {len(image_files)} 张图片")
    print("-" * 50)

    for img_path in sorted(image_files):
        # 确定输出文件名和格式
        if output_ext:
            out_name = img_path.stem + output_ext
        else:
            out_name = img_path.name

        out_path = output_dir / out_name

        # 跳过已存在文件
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            with Image.open(img_path) as img:
                # 处理模式
                if mode == "center":
                    # 如果图片小于目标尺寸，先resize到至少一边满足
                    w, h = img.size
                    if w < target_w or h < target_h:
                        ratio = max(target_w / w, target_h / h)
                        new_w = int(w * ratio)
                        new_h = int(h * ratio)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                    result = crop_center(img, target_w, target_h)

                elif mode == "resize":
                    # 直接拉伸/压缩到目标尺寸
                    result = img.resize((target_w, target_h), Image.LANCZOS)

                elif mode == "pad":
                    # 等比缩放+填充
                    result = resize_with_pad(img, target_w, target_h)

                else:
                    raise ValueError(f"未知模式: {mode}")

                # 保存（保持原格式或转换）
                save_kwargs = {}
                if out_path.suffix.lower() in {'.jpg', '.jpeg'}:
                    save_kwargs['quality'] = 95
                    # RGBA转RGB
                    if result.mode == 'RGBA':
                        result = result.convert('RGB')

                result.save(out_path, **save_kwargs)
                processed += 1

                if processed % 100 == 0:
                    print(f"  进度: {processed}/{len(image_files)}")

        except Exception as e:
            errors.append((img_path.name, str(e)))

    print(f"\n✅ 处理完成: {processed} 张")
    print(f"⏭️  跳过(已存在): {skipped} 张")

    if errors:
        print(f"\n❌ 失败 ({len(errors)}个):")
        for name, err in errors[:10]:
            print(f"   - {name}: {err}")

    return output_dir


# ==================== 执行 ====================

if __name__ == "__main__":
    result = batch_crop_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        mode="resize",  # 可选: "center", "resize", "pad"
        output_ext=OUTPUT_EXT,
        overwrite=OVERWRITE
    )
    print(f"\n📂 输出目录: {result}")