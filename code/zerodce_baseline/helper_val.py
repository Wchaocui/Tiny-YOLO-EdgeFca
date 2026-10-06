import sys, os, json
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import torch
torch.set_num_threads(3)

def main():
    w, repo, data = sys.argv[1], sys.argv[2], sys.argv[3]
    skip_val = len(sys.argv) > 4 and sys.argv[4] == "skipval"
    is_rtdetr = len(sys.argv) > 5 and sys.argv[5] == "rtdetr"
    sys.path.insert(0, repo)
    model = None
    if is_rtdetr:
        try:
            from ultralytics import RTDETR
            model = RTDETR(w)
        except Exception:
            model = None
    if model is None:
        from ultralytics import YOLO
        try:
            model = YOLO(w)
        except Exception:
            from ultralytics import RTDETR
            model = RTDETR(w)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = None
    for attempt in (False, True):
        try:
            info = model.info(verbose=attempt)
            if isinstance(info, (tuple, list)) and len(info) >= 4 and info[3]:
                gflops = round(float(info[3]), 2); break
        except Exception:
            pass
    if gflops is None:
        try:
            from ultralytics.utils.torch_utils import get_flops
            gflops = round(get_flops(model.model, 640), 2)
        except Exception:
            gflops = -1
    out = {"params_M": round(params, 2), "GFLOPs": gflops}
    if not skip_val:
        r = model.val(data=data, split="test", device="cpu", workers=2,
                      verbose=False, plots=False, save_json=False)
        b = r.box
        mAP90 = None
        try:
            aa = np.asarray(b.all_ap, dtype=float)
            if aa.ndim == 2 and aa.shape[1] >= 9:
                mAP90 = round(float(np.nanmean(aa[:, 8])), 4)
        except Exception:
            pass
        out.update({"P": round(float(b.mp), 4), "R": round(float(b.mr), 4),
                    "mAP50": round(float(b.map50), 4), "mAP75": round(float(b.map75), 4),
                    "mAP90": mAP90, "mAP50_95": round(float(b.map), 4)})
        # wtefnet-style zero guard: count detections
        try:
            import numpy as _np
            sb = getattr(r, "speed", {})
            out["n_pred_warn"] = (float(b.mp) == 0.0 and float(b.mr) == 0.0)
        except Exception:
            pass
    print("RESULT_JSON:" + json.dumps(out))

if __name__ == "__main__":
    main()
