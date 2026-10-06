from ultralytics import YOLO

# ===================== 配置项（根据自己路径修改）=====================
WEIGHT_PATH = "../runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"  # 原始FP32权重
DATA_YAML = "your_dataset.yaml"       # 你的数据集配置文件(必填，INT8校准用)
IMAGE_SIZE = 640                      # 输入分辨率
DEVICE = 0                            # 使用GPU 0
WORKSPACE = 2                         # TensorRT 显存占用(GB)
# ====================================================================

# 加载模型
model = YOLO(WEIGHT_PATH)

# ---------------------- 方式1：导出 INT8 TensorRT 量化模型（核心）----------------------
print("===== Start exporting INT8 TensorRT engine =====")
model.export(
    format="engine",
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    workspace=WORKSPACE,
    int8=True,        # 开启INT8量化 PTQ
    data=DATA_YAML,   # 校准数据集，必须填写
    simplify=True
)
print("INT8 engine export finished!")

# ---------------------- 可选：顺带导出 FP16 / ONNX 做对比（保留你原有逻辑）----------------------
# 导出 FP16 TensorRT
print("\n===== Start exporting FP16 TensorRT engine =====")
model.export(
    format="engine",
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    workspace=WORKSPACE,
    half=True,
    simplify=True
)

# 导出 ONNX（你原有代码）
# print("\n===== Start exporting ONNX =====")
# model.export(
#     format="onnx",
#     imgsz=IMAGE_SIZE,
#     half=True,
#     simplify=True,
#     device=DEVICE
# )
