import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_brightness(img):
    """计算图像区域的平均亮度"""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return np.mean(gray)


def calculate_noise(img):
    """计算图像区域的噪声水平（使用标准差法）"""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    return np.std(noise)


def analyze_images_from_labels(image_dir, label_dir, output_csv):
    """
    根据标签文件分析图像，计算目标区域亮度和整图噪声

    参数:
    image_dir: 图像文件夹路径
    label_dir: 标签文件夹路径
    output_csv: 输出CSV文件路径
    """
    results = []

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 处理每个图像
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"警告: 找不到图像 {image_file} 对应的标签文件")
            continue

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像 {image_file}")
            continue

        img_height, img_width = img.shape[:2]

        # 计算整图噪声
        whole_image_noise = calculate_noise(img)

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()


        # 处理每个目标
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # 至少需要类别和边界框信息
                continue

            # 解析YOLO格式标签
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 计算边界框在原图中的坐标
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)

            # 确保边界框坐标在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # 提取目标区域
            if x_min >= x_max or y_min >= y_max:
                continue

            obj_region = img[y_min:y_max, x_min:x_max]
            if obj_region.size == 0:
                continue

            # 计算目标区域亮度
            brightness = calculate_brightness(obj_region)

            # 添加到结果列表
            results.append({
                'image_name': image_file,
                'class_id': class_id,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'brightness': brightness,
                'whole_image_noise': whole_image_noise
            })

    # 保存结果到CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"分析完成，结果已保存到 {output_csv}")
        print(f"共处理 {len(image_files)} 张图像，分析了 {len(results)} 个目标")
    else:
        print("没有找到可分析的目标")


def analyze_confidence_intervals(csv_path, output_path=None):
    """
    根据置信度区间分析图像的平均亮度和平均噪声值

    参数:
    csv_path: 输入CSV文件路径
    output_path: 输出CSV文件路径，默认为"confidence_analysis.csv"

    返回:
    分析结果DataFrame
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 {csv_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 读取CSV文件时发生异常: {e}")
        return None

    # 检查必要的列是否存在
    required_columns = ['brightness', 'whole_image_noise', 'confidence']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少必要的列 '{col}'")
            return None

    # 处理缺失值：将confidence列中的空值或NaN替换为0
    df['confidence'] = df['confidence'].fillna(0)

    # 定义置信度区间
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

    # 将置信度值分箱
    df['confidence_interval'] = pd.cut(df['confidence'], bins=bins, labels=labels, right=False)

    # 按置信度区间分组，计算每个区间的平均亮度和平均噪声值
    grouped = df.groupby('confidence_interval').agg({
        'brightness': 'mean',
        'whole_image_noise': 'mean',
        'confidence': 'count'  # 计算每个区间的样本数
    }).reset_index()

    # 重命名列
    grouped.columns = ['置信度区间', '平均亮度', '平均噪声值', '样本数']

    # 如果没有指定输出路径，使用默认名称
    if output_path is None:
        base_name, ext = os.path.splitext(csv_path)
        output_path = f"{base_name}_analysis{ext}"

    # 保存分析结果
    try:
        grouped.to_csv(output_path, index=False)
        print(f"分析结果已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存CSV文件时发生异常: {e}")

    return grouped


def count_bin():
    # 使用示例
    input_csv = "combined_results.csv"  # 替换为实际的CSV文件路径
    output_csv = None  # 可以指定输出路径，或设为None使用默认值

    analysis_result = analyze_confidence_intervals(input_csv, output_csv)

    if analysis_result is not None:
        print("\n分析结果:")
        print(analysis_result)



from io import BytesIO



def analyze_brightness_confidence(csv_path, output_img_path=None):
    """
    分析不同亮度区间下置信度的分布并绘制混淆矩阵

    参数:
    csv_path: 输入CSV文件路径
    output_img_path: 输出图像路径，默认为"brightness_confidence_matrix.png"

    返回:
    混淆矩阵数据和图像的Base64编码（用于在网页中显示）
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 {csv_path} 不存在")
        return None, None
    except Exception as e:
        print(f"错误: 读取CSV文件时发生异常: {e}")
        return None, None

    # 检查必要的列是否存在
    required_columns = ['brightness', 'confidence']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少必要的列 '{col}'")
            return None, None

    # 处理confidence列中的空值：明确将NaN替换为0
    df['confidence'] = df['confidence'].fillna(0)

    # 处理brightness列中的空值（如果有）
    df['brightness'] = df['brightness'].fillna(0)

    # 定义亮度区间（0-225，步长15）
    brightness_bins = list(range(0, 240, 15))
    brightness_labels = [f"{i}-{i + 15}" for i in brightness_bins[:-1]]

    # 定义置信度区间（0-1，步长0.1）
    confidence_bins = np.round(np.arange(0.0, 1.1, 0.1), 1)
    confidence_labels = [f"{i:.1f}-{i + 0.1:.1f}" for i in confidence_bins[:-1]]

    # 将亮度和置信度分箱
    df['brightness_interval'] = pd.cut(df['brightness'], bins=brightness_bins,
                                       labels=brightness_labels, right=False)
    df['confidence_interval'] = pd.cut(df['confidence'], bins=confidence_bins,
                                       labels=confidence_labels, right=False)

    # 创建混淆矩阵（亮度区间 x 置信度区间）
    confusion_matrix = pd.crosstab(df['brightness_interval'], df['confidence_interval'])

    # 确保所有区间都存在于结果中
    for conf_label in confidence_labels:
        if conf_label not in confusion_matrix.columns:
            confusion_matrix[conf_label] = 0

    # 按亮度区间和置信度区间排序
    confusion_matrix = confusion_matrix.reindex(brightness_labels).sort_index(axis=1)

    # 处理混淆矩阵中的NaN值：将NaN替换为0（表示该区间没有样本）
    confusion_matrix = confusion_matrix.fillna(0)

    # 将所有值转换为整数（避免浮点数格式错误）
    confusion_matrix = confusion_matrix.astype(int)

    # 绘制混淆矩阵热图
    plt.figure(figsize=(15, 10))
    sns.set(font_scale=1.2)

    # 使用 fmt='d' 因为我们已经将数据转换为整数
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu',
                     cbar_kws={'label': 'sample number'})


    plt.xlabel('confidence', fontsize=14)
    plt.ylabel('brightness', fontsize=14)

    # 旋转x轴标签以便更好显示
    plt.xticks(rotation=45)

    # 如果没有指定输出路径，使用默认名称
    if output_img_path is None:
        output_img_path = "brightness_confidence_matrix.png"

    # 保存图像
    try:
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300)
        print(f"混淆矩阵图像已保存到: {output_img_path}")
    except Exception as e:
        print(f"错误: 保存图像时发生异常: {e}")

    # 将图像转换为Base64编码以便在网页中显示
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()

    return confusion_matrix, img_base64


def plot_confusion_matrix():
    # 使用示例
    input_csv = "combined_results.csv"  # 替换为实际的CSV文件路径
    output_img = "brightness_confidence_matrix.png"  # 输出图像路径

    matrix, img_data = analyze_brightness_confidence(input_csv, output_img)

    if matrix is not None:
        print("\n亮度-置信度分布矩阵:")
        print(matrix)

        # 如果需要在Jupyter Notebook中显示图像
        try:
            from IPython.display import Image, display
            display(Image(filename=output_img))
        except:
            print(f"图像已保存到 {output_img}，请手动打开查看")



import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64

def plot_3d_brightness_noise_confidence(csv_path, output_img_path=None,
                                        use_confidence_color=True,tick_font_size=10):
    """
    Plot 3D scatter plot of brightness, noise, and confidence with a transparent plane at z=0.8

    Parameters:
    csv_path: Path to input CSV file
    output_img_path: Path to output image, default is "3d_brightness_noise_confidence.png"
    use_confidence_color: If True, color points by confidence; if False, use point density

    Returns:
    Base64 encoded image data (for display in web pages)
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the CSV file: {e}")
        return None

    # Check if required columns exist
    required_columns = ['brightness', 'whole_image_noise', 'confidence']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV file missing required column '{col}'")
            return None

    # Handle missing values: replace NaN with 0
    df['brightness'] = df['brightness'].fillna(0)
    df['whole_image_noise'] = df['whole_image_noise'].fillna(0)
    df['confidence'] = df['confidence'].fillna(0)


    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data
    x = df['brightness']
    y = df['whole_image_noise']
    z = df['confidence']

    # Determine color mapping
    if use_confidence_color:
        # Color points by confidence value
        colors = z
        cbar_label = 'Confidence'
        cmap = 'viridis'
    else:
        # Color points by density in x-y plane
        xy = np.vstack([x, y])
        colors = stats.gaussian_kde(xy)(xy)
        cbar_label = 'Point Density'
        cmap = 'viridis'

    # Plot 3D scatter plot
    scatter = ax.scatter(x, y, z, c=colors, s=50, cmap=cmap, alpha=0.7)

    # Add color bar with reduced size, adjusted position, and custom ticks
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.1, shrink=0.5)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Add a transparent plane at z=0.8
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.ones_like(xx) * 0.8  # Plane at z=0.8

    ax.plot_surface(xx, yy, zz, alpha=0.3, color='r', linestyle='-')

    # Add text annotation for the plane
    ax.text(x_min, y_min, 0.81, "z=0.8", color='red', fontsize=12)

    # Set axis labels
    ax.set_xlabel('Brightness', fontsize=13, labelpad=10)
    ax.set_ylabel('Noise', fontsize=13, labelpad=10)
    ax.set_zlabel('Confidence', fontsize=13, labelpad=10)

    # Adjust tick label font size for all axes
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    # Set viewing angle
    ax.view_init(elev=15, azim=45)

    # If no output path specified, use default name
    if output_img_path is None:
        output_img_path = "3d_brightness_noise_confidence.png"

    # Save image
    try:
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300)
        print(f"3D scatter plot saved to: {output_img_path}")
    except Exception as e:
        print(f"Error: An error occurred while saving the image: {e}")

    # Convert image to Base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()

    return img_base64

def plot_3d():
    # Example usage
    input_csv = "combined_results.csv"  # Replace with your actual CSV file path
    output_img = "3d_brightness_noise_confidence.png"  # Output image path

    # Set use_confidence_color=True to color points by confidence instead of density
    img_data = plot_3d_brightness_noise_confidence(input_csv, output_img, use_confidence_color=True,
                                                   tick_font_size=12 )

    if img_data:
        print(f"Image saved to {output_img}, please open manually to view")

        # Try to display image in Jupyter Notebook if available
        try:
            from IPython.display import Image, display
            display(Image(filename=output_img))
        except:
            pass




if __name__ == "__main__":
    # 配置路径
    image_dir = "/home/s0433/local/data/spark-2022-stream-1/images/train"  # 图像文件夹路径
    label_dir = "/home/s0433/local/data/spark-2022-stream-1/labels/train"  # YOLO标签文件夹路径
    output_csv = "analysis_results_train.csv"  # 输出CSV文件路径

    # 执行分析
    # analyze_images_from_labels(image_dir, label_dir, output_csv)#用ground_truth计算区域亮度和整个图片的噪声
    # count_bin()#统计不同置信度下的平均亮度和平均噪声
    # plot_confusion_matrix()#绘制置信度和亮度的混淆矩阵
    plot_3d()
