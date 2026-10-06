import torch
from ultralytics import YOLO

PT_WEIGHT = "/home/s0433/local/projects_py/LeYOLO-main/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"
SAVE_INT8_PATH = "tiny_m_int8.pt"
DEVICE = "cuda"

# 加载FP32模型
yolo_model = YOLO(PT_WEIGHT)
model = yolo_model.model
model.eval()
model.to(DEVICE)

# 动态INT8量化
quant_model = torch.ao.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Conv2d, torch.nn.Linear},
    dtype=torch.qint8
)

# 保存
torch.save(quant_model, SAVE_INT8_PATH)
print(f"INT8 model saved: {SAVE_INT8_PATH}")
