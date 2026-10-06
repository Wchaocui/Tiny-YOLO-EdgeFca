import sys, random
import numpy as np
import torch

yaml_name, run_name = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/hy-tmp/LeYOLO-main")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(67)

from ultralytics import YOLO
model = YOLO(f"ultralytics/cfg/cfg/{yaml_name}")
model.train(
    data="ultralytics/cfg/datasets/SLLFS.yaml",
    epochs=300, patience=300, batch=32, imgsz=640,
    cache="ram", device=0, workers=4,
    pretrained=False, optimizer="SGD", seed=67, deterministic=False,
    cos_lr=True, close_mosaic=30, amp=True, multi_scale=True, save_json=True,
    iou=0.6, max_det=300, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5.0, nbs=32,
    hsv_h=0.2, hsv_s=0.3, hsv_v=0.2, degrees=10.0, translate=0.1, scale=0.5,
    fliplr=0.2, mosaic=0.5, mixup=0.3, erasing=0.0,
    project="/hy-tmp/LeYOLO-main/runs/detect", name=run_name,
)
