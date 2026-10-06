import torch
from ultralytics import YOLO

# ==================== 配置区（自行修改路径）====================
PT_WEIGHT = "runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"
SAVE_INT8_PATH = "tiny_m_int8.pt"
DEVICE = "cuda"
# =================================================================

# 1. 加载原始FP32模型
yolo_model = YOLO(PT_WEIGHT)
model = yolo_model.model
model.eval()
model.to(DEVICE)

# 2. 开启 PyTorch 动态INT8量化
# 只对卷积、全连接做量化，其余自定义算子(DWT/DCT)保留浮点运行
quant_model = torch.ao.quantization.quantize_dynamic(
    model,
    qconfig_spec={
        torch.nn.Conv2d,
        torch.nn.Linear
    },
    dtype=torch.qint8
)

# 3. 保存量化后模型
torch.save(quant_model, SAVE_INT8_PATH)
print(f"PyTorch INT8 量化模型已保存至: {SAVE_INT8_PATH}")
