import os, json, gc, traceback
import torch
from ultralytics import YOLO

OUT = "/hy-tmp/zerodce_work/seed_eval"
os.makedirs(OUT, exist_ok=True)
DATA = "/hy-tmp/zerodce_work/SLLFS_abs.yaml"

GROUP_A = [  # cwd = /hy-tmp/ultralytics-main
    "5seed_9t","5seed_10n","5seed-11n","5seed-12n","5seed-EMV-yolov3",
    "5seed-rtdetr-r18","5seed-rtdetr-r34","5seed-wtefnet","5seed-yola",
    "67-emv","67-r34","67seed-r18","67seed-wtefnet","67-yola","67-yolo11",
    "67-yolo12","67-yolo9t","67-yolov10",
]
ZERODCE = ["zerodce-yolo11n-seed42","zerodce-yolo11n-seed5","zerodce-yolo11n-seed67"]
GROUP_B = [  # cwd = /hy-tmp/LeYOLO-main
    "5seed-le-n","5seed-le-s","5seed-m-sppf","5seed-spp-edge21","5seed-Tiny_m_topk",
    "5seed-tiny_m_topkedge31","5seed-tiny_m_topkedge32","5seed-ty-m","5seed-ty-s",
    "67seed-le-n","67-sppm-topk","67-Tiny_m_spp",
]

def eval_one(run_dir, name):
    out_json = os.path.join(OUT, name + ".json")
    if os.path.exists(out_json):
        print("skip (done)", name, flush=True); return
    try:
        model = YOLO(os.path.join(run_dir, "weights", "best.pt"))
        m = model.val(data=DATA, split="test", device=0, batch=16, imgsz=640,
                      workers=4, verbose=False, plots=False)
        b = m.box
        res = {
            "P": float(b.mp.mean()), "R": float(b.mr.mean()),
            "mAP50": float(b.map50), "mAP75": float(b.map75),
            "mAP90": float(b.ap[:, 8].mean()), "mAP50-95": float(b.map),
        }
        json.dump(res, open(out_json, "w"), indent=1)
        print("OK", name, {k: round(v, 4) for k, v in res.items()}, flush=True)
    except Exception as e:
        json.dump({"error": str(e)[:300]}, open(out_json, "w"))
        print("ERR", name, str(e)[:200], flush=True)
        traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()

if os.getcwd().startswith("/hy-tmp/ultralytics"):
    for n in GROUP_A + ZERODCE:
        d = "/hy-tmp/zerodce_work/runs/" + n if n.startswith("zerodce") else "/hy-tmp/ultralytics-main/runs/detect/" + n
        eval_one(d, n)
else:
    for n in GROUP_B:
        eval_one("/hy-tmp/LeYOLO-main/runs/detect/" + n, n)
print("EVAL_ALL_DONE", flush=True)
