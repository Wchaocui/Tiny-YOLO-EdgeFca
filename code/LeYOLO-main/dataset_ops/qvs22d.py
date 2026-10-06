import numpy as np
import cv2
import json
import os


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    参数:
        q: [qw, qx, qy, qz] 四元数 (标量部分为第一个元素)
    返回:
        R: 3x3 旋转矩阵
    """
    qw, qx, qy, qz = q

    # 单位化四元数
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-7:
        return np.eye(3)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # 计算旋转矩阵元素
    R = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

    return R


def pose_to_bbox_with_quaternion(keypoints_3d, dist_coeffs, tvec, quaternion, camera_matrix, image_width, image_height):
    """
    使用四元数表示的位姿转换为边界框
    参数:
        keypoints_3d: (N, 3) 世界坐标系中的3D关键点
        tvec: (3,) 平移向量
        quaternion: (4,) 四元数 [qw, qx, qy, qz]
        camera_matrix: (3, 3) 相机内参矩阵
        image_width: 图像宽度
        image_height: 图像高度
    返回:
        bbox: [x_min, y_min, x_max, y_max] 格式的边界框
        projected_points: 投影后的2D点
    """
    # 1. 将四元数转换为旋转矩阵
    R = quaternion_to_rotation_matrix(quaternion)

    # 2. 将旋转矩阵转换为旋转向量 (OpenCV的projectPoints需要旋转向量)
    rvec, _ = cv2.Rodrigues(R)

    # 3. 投影3D点到2D图像平面
    keypoints_2d, _ = cv2.projectPoints(
        objectPoints=keypoints_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    keypoints_2d = keypoints_2d.reshape(-1, 2)

    # 4. 过滤掉投影到图像外的点
    visible_points = [pt for pt in keypoints_2d if 0 <= pt[0] < image_width and 0 <= pt[1] < image_height]

    if not visible_points:
        return [0, 0, 0, 0], []

    visible_points = np.array(visible_points)

    # 5. 计算最小外接矩形
    x_min, y_min = np.min(visible_points, axis=0)
    x_max, y_max = np.max(visible_points, axis=0)

    # 6. 添加安全边界 (1%)
    width = x_max - x_min
    height = y_max - y_min
    margin_x = width * 0.01
    margin_y = height * 0.01

    bbox = [
        max(0, x_min - margin_x),
        max(0, y_min - margin_y),
        min(image_width, x_max + margin_x),
        min(image_height, y_max + margin_y)
    ]

    return bbox, keypoints_2d


def bbox_to_yolo_format(bbox, image_width, image_height):
    """
    将边界框转换为YOLO格式
    参数:
        bbox: [x_min, y_min, x_max, y_max]
        image_width: 图像宽度
        image_height: 图像高度
    返回:
        yolo_bbox: [center_x, center_y, width, height] 归一化值 (0-1)
    """
    x_min, y_min, x_max, y_max = bbox

    # 计算中心点坐标
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    # 计算宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 归一化
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    return [center_x, center_y, width, height]


def process_json_file(json_path, output_dir):
    """
    处理JSON文件中的所有条目，计算边界框并保存为YOLO格式
    参数:
        json_path: JSON文件路径
        output_dir: 输出目录
    """
    # 相机参数
    K = np.array([
        [2988.5795163815555, 0, 960],
        [0, 2988.3401159176124, 600],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.array([
        -0.22383016606510672,
        0.51409797089106379,
        -0.00066499611998340662,
        -0.00021404771667484594,
        -0.13124227429077406
    ], dtype=np.float64)

    # 3D 关键点（11个）
    keypoints_3d = np.array([
        [-0.37, -0.37, 0.37, 0.37, -0.37, -0.37, 0.37, 0.37, -0.5427, 0.5427, 0.305],
        [-0.385, 0.385, 0.385, -0.385, -0.264, 0.304, 0.304, -0.264, 0.4877, 0.4877, -0.579],
        [0.3215, 0.3215, 0.3215, 0.3215, 0, 0, 0, 0, 0.2535, 0.2591, 0.2515]
    ]).T  # shape (11, 3)

    # 图像尺寸
    image_width, image_height = 1920, 1200

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 处理每个条目
    results = {}
    for idx, item in enumerate(data):
        # 提取位姿参数 - 使用正确的键名
        tvec = np.array(item['r_Vo2To_vbs_true'], dtype=np.float64)
        quaternion = np.array(item['q_vbs2tango_true'], dtype=np.float64)

        # 计算边界框
        bbox, projected_points = pose_to_bbox_with_quaternion(
            keypoints_3d=keypoints_3d,
            dist_coeffs=dist_coeffs,
            tvec=tvec,
            quaternion=quaternion,
            camera_matrix=K,
            image_width=image_width,
            image_height=image_height
        )

        # 转换为YOLO格式
        yolo_bbox = bbox_to_yolo_format(bbox, image_width, image_height)

        # 获取文件名
        filename = item['filename']

        # 保存结果
        results[filename] = {
            'bbox': bbox,
            'yolo_bbox': yolo_bbox,
            'projected_points': projected_points.tolist(),
            'tvec': tvec.tolist(),
            'quaternion': quaternion.tolist()
        }

        # 创建YOLO格式的标注文件
        base_name = os.path.splitext(os.path.basename(filename))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        # YOLO格式: class_id center_x center_y width height
        # 假设类别为0 (只有一个类别)
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f"1 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

        # 每处理100个条目打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(data)} 个条目")

    # 保存汇总结果到JSON文件
    summary_path = os.path.join(output_dir, "bbox_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ===== 使用示例 =====
if __name__ == "__main__":
    # 设置JSON文件路径和输出目录
    json_path = "train.json"  # 替换为您的JSON文件路径
    output_dir = "/home/s0433/local/data/speedplusv2/labels/train"  # YOLO格式标签输出目录

    print(f"开始处理JSON文件: {json_path}")
    print(f"输出目录: {output_dir}")

    # 处理JSON文件
    results = process_json_file(json_path, output_dir)

    print(f"\n处理完成! 共处理了 {len(results)} 个条目。")
    print(f"YOLO标签已保存到: {output_dir}")
    print(f"边界框汇总已保存到: {os.path.join(output_dir, 'bbox_summary.json')}")