import random
import numpy as np
import torch
from ultralytics import YOLO

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(67)

# Zero-DCE + YOLOv11-n two-stage baseline, exact paper protocol (== runs/detect/11n args.yaml, seed 42)
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.train(
    data="/hy-tmp/SLLFS_zerodce/SLLFS_zerodce.yaml",
    epochs=300, patience=300, batch=32, imgsz=640,
    cache="ram", device=0, workers=4,
    project="/hy-tmp/zerodce_work/runs", name="zerodce-yolo11n-seed67",
    pretrained=False, optimizer="SGD", verbose=True,
    seed=67, deterministic=False, cos_lr=True, close_mosaic=30,
    amp=True, multi_scale=True, save_json=True,
    iou=0.6, max_det=300,
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    hsv_h=0.2, hsv_s=0.3, hsv_v=0.2, degrees=10.0, translate=0.1,
    scale=0.5, fliplr=0.2, mosaic=0.5, mixup=0.3, erasing=0.0,
)
model.val(data="/hy-tmp/SLLFS_zerodce/SLLFS_zerodce.yaml", split="test", device="cpu")
