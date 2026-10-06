import os
from ultralytics import YOLO


def find_low_confidence_images(model, source, confidence_threshold=0.8):
    """
    找出置信度低于指定阈值的图片名称并保存。

    Args:
        model: 已加载的 YOLO 模型。
        source (str): 图片源路径（文件夹或单个文件）。
        confidence_threshold (float): 置信度阈值，默认为0.8。
    """
    # 调用模型进行推理
    results = model.predict(source=source, device='cuda',stream=False)  # 设置 stream=False 以获取完整结果列表

    # 初始化低置信度图片列表
    low_confidence_images = []

    # 遍历推理结果
    for result in results:
        # 获取图片路径
        image_path = result.path  # 使用 result1.path 获取图片路径
        image_name = os.path.basename(image_path)  # 提取图片文件名

        # 获取检测框置信度
        if hasattr(result.boxes, 'conf'):
            confidences = result.boxes.conf.cpu().numpy()  # 假设 conf 是一个 tensor
        elif hasattr(result.boxes, 'confidence'):
            confidences = result.boxes.confidence.cpu().numpy()  # 如果属性名是 confidence
        else:
            raise AttributeError("No confidence attribute found in result1.boxes")

        # 检查是否所有检测框的置信度都低于阈值
        if all(conf < confidence_threshold for conf in confidences):
            low_confidence_images.append(image_name)  # 记录低置信度图片的文件名

    # 保存低置信度图片的文件名
    output_file = "dif/low_confidence_images6.txt"
    with open(output_file, "w") as f:
        for image_name in low_confidence_images:
            f.write(image_name + "\n")

    print(f"低置信度图片的文件名已保存到 {output_file}")


# 示例：加载模型并调用函数
model =YOLO('../runs/detect/LE_s_O(346)/weights/best.pt')  # 加载你的 YOLO 模型
source ='/home/s0433/local/projects_py/LeYOLO-main/output_6'  # 替换为你的图片源路径
find_low_confidence_images(model, source)