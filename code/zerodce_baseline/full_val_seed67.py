import sys
sys.path.insert(0, "/hy-tmp/ultralytics-main")
from ultralytics import YOLO
import numpy as np
m = YOLO("/hy-tmp/zerodce_work/runs/zerodce-yolo11n-seed67/weights/best.pt")
r = m.val(data="/hy-tmp/SLLFS_zerodce/SLLFS_zerodce.yaml", split="test", device="cpu", workers=2, verbose=False)
b = r.box
cands = {}
for attr in ["all_ap", "ap", "ap_class_index"]:
    v = getattr(b, attr, None)
    try:
        arr = v() if callable(v) else v
        cands[attr] = None if arr is None else np.asarray(arr).shape
    except Exception as e:
        cands[attr] = f"ERR {e}"
print("VAL2_DONE", cands)
try:
    aa = np.asarray(b.all_ap() if callable(getattr(b, "all_ap", None)) else b.all_ap)
    print(f"mAP90={aa[:, 8].mean():.4f}")
except Exception:
    try:
        aa = np.asarray(b.ap)
        print(f"mAP90={aa[:, 8].mean():.4f}")
    except Exception as e:
        print("mAP90 unavailable:", e)
print(f"P={b.mp:.4f} R={b.mr:.4f} mAP50={b.map50:.4f} mAP75={b.map75:.4f} mAP50-95={b.map:.4f}")
