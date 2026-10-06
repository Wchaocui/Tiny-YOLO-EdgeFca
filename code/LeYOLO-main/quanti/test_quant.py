import torch
from ultralytics import YOLO

IMG_SIZE = 640
DEVICE = "cuda"
FP32_PATH = "/home/s0433/local/projects_py/LeYOLO-main/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"
INT8_PATH = "tiny_m_int8.pt"

# 加载模型
model_fp32 = YOLO(FP32_PATH).model.eval().to(DEVICE)
model_int8 = torch.load(INT8_PATH, map_location=DEVICE).eval()

# 单张图片推理示例（替换成你的图片路径）
img_path = "test.jpg"
results_fp32 = YOLO(FP32_PATH)(img_path)
results_int8 = YOLO(model_int8)(img_path)
