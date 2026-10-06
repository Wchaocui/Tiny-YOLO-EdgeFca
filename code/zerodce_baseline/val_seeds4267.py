import subprocess, csv, time, os, json, sys

PY = "/hy-tmp/envs/torch310/bin/python"
HELPER = "/hy-tmp/zerodce_work/helper_val.py"
UM = "/hy-tmp/ultralytics-main"
LM = "/hy-tmp/LeYOLO-main"
DATA = f"{UM}/ultralytics/cfg/datasets/SLLFS.yaml"
OUT = "/hy-tmp/zerodce_work/val_seeds4267_results.csv"

ENTRIES = [
    (f"{UM}/runs/detect/9t/weights/yolov9_slli.pt",           UM, "YOLOv9-t",            "42"),
    (f"{UM}/runs/detect/10n/weights/yolov10_slli.pt",         UM, "YOLOv10-n",           "42"),
    (f"{UM}/runs/detect/11n/weights/yolo11_slli.pt",          UM, "YOLOv11-n",           "42"),
    (f"{UM}/runs/detect/12n/weights/yolo12_slli.pt",          UM, "YOLOv12-n",           "42"),
    (f"{UM}/runs/detect/detr-t18/weights/r18_slli.pt",        UM, "RT-DETR-R18",         "42"),
    (f"{UM}/runs/detect/rtdetr-r34/weights/r34_slli.pt",      UM, "RT-DETR-R34",         "42"),
    (f"{UM}/runs/detect/EMV-YOLOV3/weights/best.pt",          UM, "EMV-YOLOv3-tiny",     "42"),
    (f"{UM}/runs/detect/wtefnet/weights/best.pt",             UM, "WTEFNet",             "42"),
    (f"{UM}/runs/detect/yola/weights/best.pt",                UM, "YOLA",                "42"),
    (f"{LM}/runs/detect/leyolo_n/weights/leyolo_n_slli.pt",   LM, "LeYOLO-nano",         "42"),
    (f"{LM}/runs/detect/leyolo_s/weights/leyolo_s_slli.pt",   LM, "LeYOLO-small",        "42"),
    (f"{LM}/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt", LM, "Tiny-YOLO-EdgeFca_s", "42"),
    (f"{LM}/runs/detect/Tinyolo_s_topkedge21/weights/tiny_s_slli.pt", LM, "Tiny-YOLO-EdgeFca_n", "42"),
    (f"{UM}/runs/detect/67-yolo9t/weights/best.pt",           UM, "YOLOv9-t",            "67"),
    (f"{UM}/runs/detect/67-yolov10/weights/best.pt",          UM, "YOLOv10-n",           "67"),
    (f"{UM}/runs/detect/67-yolo11/weights/best.pt",           UM, "YOLOv11-n",           "67"),
    (f"{UM}/runs/detect/67-yolo12/weights/best.pt",           UM, "YOLOv12-n",           "67"),
    (f"{UM}/runs/detect/67seed-r18/weights/best.pt",          UM, "RT-DETR-R18",         "67"),
    (f"{UM}/runs/detect/67-r34/weights/best.pt",              UM, "RT-DETR-R34",         "67"),
    (f"{UM}/runs/detect/67-emv/weights/best.pt",              UM, "EMV-YOLOv3-tiny",     "67"),
    (f"{UM}/runs/detect/67seed-wtefnet/weights/best.pt",      UM, "WTEFNet",             "67"),
    (f"{UM}/runs/detect/67-yola/weights/best.pt",             UM, "YOLA",                "67"),
]

def run_one(w, repo):
    env = dict(os.environ); env["PYTHONPATH"] = repo
    p = subprocess.run([PY, HELPER, w, repo, DATA], capture_output=True, text=True, env=env, cwd="/", timeout=7200)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError((p.stderr or p.stdout)[-200:])

def main():
    rows, t0 = [], time.time()
    for i, (w, repo, name, seed) in enumerate(ENTRIES):
        try:
            rec = run_one(w, repo)
            rec.update({"paper_name": name, "seed": seed})
            rows.append(rec)
            print(f"[{i+1}/{len(ENTRIES)}] {name} s{seed}: {rec} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(ENTRIES)}] {name} s{seed} FAILED: {str(e)[:150]}", flush=True)
            rows.append({"paper_name": name, "seed": seed, "error": str(e)[:120]})
        with open(OUT, "w", newline="") as f:
            fw = csv.DictWriter(f, fieldnames=["paper_name", "seed", "params_M", "GFLOPs",
                                               "P", "R", "mAP50", "mAP75", "mAP90", "mAP50_95", "error"])
            fw.writeheader(); fw.writerows(rows)
    print("ALL_4267_VAL_DONE", flush=True)

if __name__ == "__main__":
    main()
