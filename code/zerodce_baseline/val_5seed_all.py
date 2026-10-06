import os, sys, csv, json, time
import numpy as np
import torch
torch.set_num_threads(6)

ENTRIES = [
    # (run_dir, sys_path_repo, paper_name)
    ("/hy-tmp/ultralytics-main/runs/detect/5seed_9t",           "/hy-tmp/ultralytics-main", "YOLOv9-t"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed_10n",          "/hy-tmp/ultralytics-main", "YOLOv10-n"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed10n-2",         "/hy-tmp/ultralytics-main", "YOLOv10-n(rep)"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-11n",          "/hy-tmp/ultralytics-main", "YOLOv11-n"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-12n",          "/hy-tmp/ultralytics-main", "YOLOv12-n"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-rtdetr-r18",   "/hy-tmp/ultralytics-main", "RT-DETR-R18"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-rtdetr-r34",   "/hy-tmp/ultralytics-main", "RT-DETR-R34"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-EMV-yolov3",   "/hy-tmp/ultralytics-main", "EMV-YOLOv3-tiny"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-wtefnet",      "/hy-tmp/ultralytics-main", "WTEFNet"),
    ("/hy-tmp/ultralytics-main/runs/detect/5seed-yola",         "/hy-tmp/ultralytics-main", "YOLA"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-le-n",              "/hy-tmp/LeYOLO-main",      "LeYOLO-nano"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-le-s",              "/hy-tmp/LeYOLO-main",      "LeYOLO-small"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-m",              "/hy-tmp/LeYOLO-main",      "Tiny-YOLO-EdgeFca_s"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-s",              "/hy-tmp/LeYOLO-main",      "Tiny-YOLO-EdgeFca_n"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-m-sppf",            "/hy-tmp/LeYOLO-main",      "[TableIV] SPPF baseline"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-spp-edge21",        "/hy-tmp/LeYOLO-main",      "[TableIV] SPPF+EdgeFca"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-Tiny_m_topk",       "/hy-tmp/LeYOLO-main",      "[TableIV] MMA w/o EdgeFca"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-tiny_m_topkedge31", "/hy-tmp/LeYOLO-main",      "[TableIV] edge31"),
    ("/hy-tmp/LeYOLO-main/runs/detect/5seed-tiny_m_topkedge32", "/hy-tmp/LeYOLO-main",      "[TableIV] edge32"),
]
DATA = "/hy-tmp/ultralytics-main/ultralytics/cfg/datasets/SLLFS.yaml"
OUT_CSV = "/hy-tmp/zerodce_work/val_5seed_all_results.csv"

def val_one(run, repo, name):
    sys.path.insert(0, repo)
    for m in list(sys.modules):
        if m.startswith("ultralytics"):
            del sys.modules[m]
    from ultralytics import YOLO
    w = os.path.join(run, "weights", "best.pt")
    model = YOLO(w)
    r = model.val(data=DATA, split="test", device="cpu", workers=2, verbose=False, plots=False)
    b = r.box
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    mAP90 = float(np.asarray(b.all_ap)[:, 8].mean())
    rec = {"run": os.path.basename(run), "paper_name": name, "params_M": round(params, 2),
           "P": round(float(b.mp), 4), "R": round(float(b.mr), 4),
           "mAP50": round(float(b.map50), 4), "mAP75": round(float(b.map75), 4),
           "mAP90": round(mAP90, 4), "mAP50_95": round(float(b.map), 4)}
    del model
    torch.cuda.empty_cache()
    return rec

def main():
    rows = []
    t0 = time.time()
    for i, (run, repo, name) in enumerate(ENTRIES):
        try:
            rec = val_one(run, repo, name)
            rows.append(rec)
            print(f"[{i+1}/{len(ENTRIES)}] {name} ({rec['run']}): P={rec['P']} R={rec['R']} "
                  f"mAP50={rec['mAP50']} mAP75={rec['mAP75']} mAP90={rec['mAP90']} mAP50-95={rec['mAP50_95']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(ENTRIES)}] {name} FAILED: {e}", flush=True)
            rows.append({"run": os.path.basename(run), "paper_name": name, "error": str(e)[:120]})
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "paper_name", "params_M", "P", "R",
                                              "mAP50", "mAP75", "mAP90", "mAP50_95", "error"])
            w.writeheader(); w.writerows(rows)
    print("ALL_5SEED_VAL_DONE", flush=True)

if __name__ == "__main__":
    main()
