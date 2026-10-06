import sys, os, json, gc
import numpy as np
REPO = sys.argv[1]
sys.path.insert(0, REPO)
import torch
from ultralytics import YOLO

OUT = "/hy-tmp/zerodce_work/seed_eval"
os.makedirs(OUT, exist_ok=True)
DATA = "/hy-tmp/zerodce_work/SLLFS_abs.yaml"

RUNS = [
    "/hy-tmp/ultralytics-main/runs/detect/5seed_9t",
    "/hy-tmp/ultralytics-main/runs/detect/5seed_10n",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-11n",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-12n",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-rtdetr-r18",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-rtdetr-r34",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-EMV-yolov3",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-wtefnet",
    "/hy-tmp/ultralytics-main/runs/detect/5seed-yola",
    "/hy-tmp/LeYOLO-main/runs/detect/5seed-le-n",
    "/hy-tmp/LeYOLO-main/runs/detect/5seed-le-s",
    "/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-m",
    "/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-s",
]

def ap_matrix(b):
    for attr in ("all_ap", "ap"):
        a = getattr(b, attr, None)
        if a is not None:
            a = np.asarray(a)
            if a.ndim == 2 and a.shape[1] == 10:
                return a
    raise RuntimeError("no ap matrix found")

for d in RUNS:
    name = os.path.basename(d)
    out_json = os.path.join(OUT, name + ".json")
    if os.path.exists(out_json):
        try:
            if "error" not in json.load(open(out_json)):
                print("skip", name, flush=True); continue
        except Exception:
            pass
    try:
        model = YOLO(os.path.join(d, "weights", "best.pt"))
        m = model.val(data=DATA, split="test", device=0, batch=16, imgsz=640,
                      workers=4, verbose=False, plots=False)
        b = m.box
        res = {
            "P": float(np.mean(b.mp)), "R": float(np.mean(b.mr)),
            "mAP50": float(b.map50), "mAP75": float(b.map75),
            "mAP90": float(ap_matrix(b)[:, 8].mean()), "mAP50-95": float(b.map),
        }
        json.dump(res, open(out_json, "w"), indent=1)
        print("OK", name, {k: round(v, 4) for k, v in res.items()}, flush=True)
    except Exception as e:
        json.dump({"error": str(e)[:300]}, open(out_json, "w"))
        print("ERR", name, str(e)[:200], flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
print("SEED5_DONE", flush=True)
