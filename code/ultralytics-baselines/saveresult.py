import os
import torch
import pandas as pd
from ultralytics import YOLO


def yolo_results_to_csv(model_path, image_dir, csv_path, device='cuda'):
    """
    将YOLO模型的推理结果按照图片名称保存到CSV文件

    参数:
    model_path: YOLO模型路径
    image_dir: 图像文件夹路径
    csv_path: 输出CSV文件路径
    device: 推理设备，'cuda'或'cpu'
    """
    # 检查设备是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU进行推理")
        device = 'cpu'

    # 加载模型
    model = YOLO(model_path)

    # 存储所有结果的数据列表
    all_results = []

    # 遍历图像文件夹
    for img_name in os.listdir(image_dir):
        # 只处理图像文件
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue

        img_path = os.path.join(image_dir, img_name)

        # 模型推理
        results = model.predict(img_path, save=False, device=device)

        # 处理每个检测结果
        for result in results:
            boxes = result.boxes  # 检测框信息

            for box in boxes:
                cls = int(box.cls)  # 类别ID
                conf = float(box.conf)  # 置信度
                xyxy = box.xyxy[0].tolist()  # 边界框坐标(x1, y1, x2, y2)

                # 添加到结果列表
                all_results.append({
                    'image_name': img_name,
                    'class_id': cls,
                    'class_name': model.names[cls],  # 类别名称
                    'confidence': conf,
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3]
                })

    # 创建DataFrame并保存到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        print(f"结果已保存到 {csv_path}")
        print(f"共处理 {len(os.listdir(image_dir))} 张图像，检测到 {len(all_results)} 个目标")
    else:
        print("没有检测到任何目标，CSV文件未生成")


import pandas as pd
import os


def process_yolo_results(csv_path, output_path=None):
    """
    处理YOLOv8检测结果CSV文件，对同一图片的多个检测结果进行合并

    参数:
    csv_path: 输入CSV文件路径
    output_path: 输出CSV文件路径，默认在原文件名后添加"_processed"

    返回:
    处理后的DataFrame
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
    required_columns = ['image_name', 'confidence']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少必要的列 '{col}'")
            return None

    # 按图片名称分组，找出每个图片的检测结果数
    image_counts = df['image_name'].value_counts()

    # 找出有多个检测结果的图片
    images_with_multiple_results = image_counts[image_counts > 1].index.tolist()

    print(f"总图片数: {len(image_counts)}")
    print(f"有多个检测结果的图片数: {len(images_with_multiple_results)}")

    # 创建一个新的DataFrame来存储处理后的结果
    processed_df = pd.DataFrame(columns=df.columns)

    # 处理每个图片
    for image_name in image_counts.index:
        image_rows = df[df['image_name'] == image_name].copy()

        if image_name in images_with_multiple_results:
            # 对于有多个结果的图片，将置信度改为1/结果数
            result_count = len(image_rows)
            image_rows['confidence'] = 1.0 / result_count

            # 只保留第一条结果
            processed_df = pd.concat([processed_df, image_rows.iloc[0:1]])
        else:
            # 对于只有一个结果的图片，直接保留
            processed_df = pd.concat([processed_df, image_rows])

    # 如果没有指定输出路径，默认在原文件名后添加"_processed"
    if output_path is None:
        base_name, ext = os.path.splitext(csv_path)
        output_path = f"{base_name}_processed{ext}"

    # 保存处理后的结果
    try:
        processed_df.to_csv(output_path, index=False)
        print(f"处理后的结果已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存CSV文件时发生异常: {e}")

    return processed_df


def adjust():
    # 使用示例
    input_csv = "yolov8_results_train.csv"  # 替换为实际的CSV文件路径
    output_csv = "yolov8_results_train_ad.csv"  # 可以指定输出路径，或设为None使用默认值

    processed_data = process_yolo_results(input_csv, output_csv)

    if processed_data is not None:
        print("\n处理结果概览:")
        print(processed_data.head())


import pandas as pd
import os


def merge_csv_files(file1_path, file2_path, output_path=None, on_column='image_name'):
    """
    以第一个CSV文件为基准，按指定列合并两个CSV文件

    参数:
    file1_path: 第一个CSV文件路径（基准文件）
    file2_path: 第二个CSV文件路径
    output_path: 输出文件路径，默认为"merged.csv"
    on_column: 用于匹配的列名，默认为"image_name"

    返回:
    合并后的DataFrame
    """
    # 读取两个CSV文件
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e.filename}")
        return None
    except Exception as e:
        print(f"错误: 读取CSV文件时发生异常 - {e}")
        return None

    print(f"第一个文件: {len(df1)} 行")
    print(f"第二个文件: {len(df2)} 行")

    # 检查两个文件是否都包含用于匹配的列
    if on_column not in df1.columns or on_column not in df2.columns:
        print(f"错误: 两个文件中必须都包含用于匹配的列 '{on_column}'")
        return None

    # 以第一个文件为基准进行左连接
    merged_df = pd.merge(df1, df2, on=on_column, how='left', suffixes=('_file1', '_file2'))

    print(f"合并后文件: {len(merged_df)} 行")

    # 如果没有指定输出路径，使用默认名称
    if output_path is None:
        output_path = "merged.csv"

    # 保存合并后的文件
    try:
        merged_df.to_csv(output_path, index=False)
        print(f"合并后的文件已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存CSV文件时发生异常 - {e}")

    return merged_df


def merge_files():
    # 使用示例
    file1 = "analysis_results_train.csv"  # 第一个CSV文件路径（基准）
    file2 = "yolov8_results_train_ad.csv"  # 第二个CSV文件路径
    output_file = "merged_train.csv"  # 输出文件路径

    merged_data = merge_csv_files(file1, file2, output_file)

    if merged_data is not None:
        print("\n合并结果概览:")
        print(merged_data.head())


def concatenate_csv_files(file1_path, file2_path, file3_path, output_path=None):
    """
    按行连接（纵向堆叠）三个CSV文件

    参数:
    file1_path: 第一个CSV文件路径
    file2_path: 第二个CSV文件路径
    file3_path: 第三个CSV文件路径
    output_path: 输出文件路径，默认为"concatenated.csv"

    返回:
    合并后的DataFrame
    """
    # 读取三个CSV文件
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        df3 = pd.read_csv(file3_path)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e.filename}")
        return None
    except Exception as e:
        print(f"错误: 读取CSV文件时发生异常 - {e}")
        return None

    # 检查三个文件是否有相同的列
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    columns3 = set(df3.columns)

    if columns1 != columns2 or columns1 != columns3:
        print("警告: 三个文件的列不完全相同")
        print(f"文件1列数: {len(columns1)}, 文件2列数: {len(columns2)}, 文件3列数: {len(columns3)}")
        print(f"文件1列: {list(columns1)}")
        print(f"文件2列: {list(columns2)}")
        print(f"文件3列: {list(columns3)}")

        # 获取所有列的并集
        all_columns = columns1.union(columns2, columns3)

        # 确保所有DataFrame具有相同的列
        df1 = df1.reindex(columns=all_columns)
        df2 = df2.reindex(columns=all_columns)
        df3 = df3.reindex(columns=all_columns)

    # 按行连接三个DataFrame
    concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)

    # 验证合并后的行数
    expected_rows = len(df1) + len(df2) + len(df3)
    actual_rows = len(concatenated_df)

    if expected_rows != actual_rows:
        print(f"警告: 合并后的行数 ({actual_rows}) 不等于预期的行数 ({expected_rows})")
    else:
        print(f"合并成功: 总共有 {actual_rows} 行")

    # 如果没有指定输出路径，使用默认名称
    if output_path is None:
        output_path = "concatenated.csv"

    # 保存合并后的文件
    try:
        concatenated_df.to_csv(output_path, index=False)
        print(f"合并后的文件已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 保存CSV文件时发生异常 - {e}")

    return concatenated_df


def concat_files():
    # 使用示例
    file1 = "merged_train.csv"  # 第一个CSV文件路径 (660行)
    file2 = "merged_test.csv"  # 第二个CSV文件路径 (110行)
    file3 = "merged_val.csv"  # 第三个CSV文件路径 (110行)
    output_file = "combined_results.csv"  # 输出文件路径

    combined_data = concatenate_csv_files(file1, file2, file3, output_file)

    if combined_data is not None:
        print("\n合并结果概览:")
        print(combined_data.head())



if __name__ == "__main__":
    # 配置参数
    model_path = "runs/detect/predatatestv8/weights/best.pt"  # YOLO模型路径
    image_dir = "/home/s0433/local/data/spark-2022-stream-1/images/val"  # 图像文件夹路径
    csv_path = "yolov8_results_val.csv"  # 输出CSV文件路径
    device = "cuda"  # 推理设备，'cuda'或'cpu'

    # 执行转换
    #yolo_results_to_csv(model_path, image_dir, csv_path, device)
    #adjust()#去掉一张图片重复的检测条目，将置信度置为1/重复数
    # merge_files()#将train/val/test的csv按照行分别合并
    concat_files()
