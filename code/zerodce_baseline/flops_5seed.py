import os, sys, csv
import torch
UM, LM = "/hy-tmp/ultralytics-main", "/hy-tmp/LeYOLO-main"
CKPTS = [
    ("5seed_9t",           f"{UM}/runs/detect/5seed_9t/weights/best.pt",           UM),
    ("5seed_10n",          f"{UM}/runs/detect/5seed_10n/weights/best.pt",          UM),
    ("5seed-11n",          f"{UM}/runs/detect/5seed-11n/weights/best.pt",          UM),
    ("5seed-12n",          f"{UM}/runs/detect/5seed-12n/weights/best.pt",          UM),
    ("5seed-rtdetr-r18",   f"{UM}/runs/detect/5seed-rtdetr-r18/weights/best.pt",   UM),
    ("5seed-rtdetr-r34",   f"{UM}/runs/detect/5seed-rtdetr-r34/weights/best.pt",   UM),
    ("5seed-EMV",          f"{UM}/runs/detect/5seed-EMV-yolov3/weights/best.pt",   UM),
    ("5seed-wtefnet",      f"{UM}/runs/detect/5seed-wtefnet/weights/best.pt",      UM),
    ("5seed-yola",         f"{UM}/runs/detect/5seed-yola/weights/best.pt",         UM),
    ("5seed-le-n",         f"{LM}/runs/detect/5seed-le-n/weights/best.pt",         LM),
    ("5seed-le-s",         f"{LM}/runs/detect/5seed-le-s/weights/best.pt",         LM),
    ("5seed-ty-m(s)",      f"{LM}/runs/detect/5seed-ty-m/weights/best.pt",         LM),
    ("5seed-ty-s(n)",      f"{LM}/runs/detect/5seed-ty-s/weights/best.pt",         LM),
]
rows = []
for name, w, repo in CKPTS:
    try:
        sys.path.insert(0, repo)
        for m in list(sys.modules):
            if m.startswith("ultralytics"):
                del sys.modules[m]
        from ultralytics import YOLO
        model = YOLO(w)
        info = model.info(verbose=False)
        rows.append({"run": name, "params_M": round(info[1]/1e6, 2), "GFLOPs": round(info[3], 2)})
        print(name, rows[-1], flush=True)
        del model
    except Exception as e:
        print(name, "FAILED", str(e)[:100], flush=True)
with open("/hy-tmp/zerodce_work/flops_5seed.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=["run", "params_M", "GFLOPs"]); w_.writeheader(); w_.writerows(rows)
print("FLOPS_DONE", flush=True)
