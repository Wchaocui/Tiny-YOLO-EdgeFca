import sys
sys.path.insert(0, "/hy-tmp/ultralytics-main")
from ultralytics import YOLO
m = YOLO("/hy-tmp/zerodce_work/runs/zerodce-yolo11n-seed42/weights/best.pt")
r = m.val(data="/hy-tmp/SLLFS_zerodce/SLLFS_zerodce.yaml", split="test", device="cpu", workers=2, verbose=True)
import numpy as np
ap = r.box.ap  # nc x 10
print("VAL_DONE")
print(f"P={r.box.mp:.4f} R={r.box.mr:.4f} mAP50={r.box.map50:.4f} mAP75={r.box.map75:.4f}")
print(f"mAP90={ap[:, 8].mean():.4f} mAP50-95={r.box.map:.4f}")
