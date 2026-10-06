import json, os, glob
import numpy as np
from PIL import Image

IMG = "/hy-tmp/SLLFS/images"
LBL = "/hy-tmp/SLLFS/labels"
idx, lidx = {}, {}
for s in ["train", "val", "test"]:
    for f in glob.glob(os.path.join(IMG, s, "*.jpg")):
        idx[os.path.basename(f)] = f
        lp = os.path.join(LBL, s, os.path.splitext(os.path.basename(f))[0] + ".txt")
        lidx[os.path.basename(f)] = lp if os.path.exists(lp) else None

def boxes(name):
    p = lidx[name]
    out = []
    if p:
        for line in open(p):
            parts = line.split()
            if len(parts) >= 5:
                out.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2
    bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    ix = max(0, min(ax2,bx2)-max(ax1,bx1)); iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy; ua = a[3]*a[4]+b[3]*b[4]-inter
    return inter/ua if ua > 0 else 0

def crop_norm(path, box, size=64, pad=0.1):
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    H, W = img.shape
    x1 = max(0, int((box[1]-box[3]/2-pad)*W)); y1 = max(0, int((box[2]-box[4]/2-pad)*H))
    x2 = min(W, int((box[1]+box[3]/2+pad)*W)); y2 = min(H, int((box[2]+box[4]/2+pad)*H))
    c = img[y1:y2, x1:x2]
    c = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((size, size)), dtype=np.float32)
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    return (c - lo) / max(hi - lo, 1e-3)

def pair_score(na, nb):
    ba, bb = boxes(na), boxes(nb)
    if not ba or not bb: return None
    best = max(((iou(a, b), a, b) for a in ba for b in bb), key=lambda t: t[0])
    io, a, b = best
    if a[0] != b[0]: return {"same_class": False, "iou": round(io, 3), "crop_mad": None}
    if io < 0.3: return {"same_class": True, "iou": round(io, 3), "crop_mad": None}
    ca, cb = crop_norm(idx[na], a), crop_norm(idx[nb], b)
    return {"same_class": True, "iou": round(io, 3), "crop_mad": round(float(np.abs(ca - cb).mean()) * 255, 1)}

d = json.load(open("dup_scan_result.json"))
cross = d["cross"]
rng = np.random.default_rng(1)
sample = rng.choice(len(cross), size=40, replace=False)
names = sorted(idx)
print("=== sampled CROSS-SPLIT flagged pairs (target-level) ===")
n_dup = 0
for i in sample:
    a, b, hdist, sa, sb = cross[i]
    r = pair_score(a, b)
    if r and r["same_class"] and r["iou"] >= 0.5 and r["crop_mad"] is not None and r["crop_mad"] < 15:
        n_dup += 1
    print(f"  {a}|{b}: {r}")
print(f"genuine target-level near-dups in cross sample: {n_dup}/40")
print("=== baseline RANDOM pairs ===")
n_dup_r = 0
for _ in range(40):
    a, b = rng.choice(names, 2, replace=False)
    r = pair_score(a, b)
    if r and r["same_class"] and r["iou"] >= 0.5 and r["crop_mad"] is not None and r["crop_mad"] < 15:
        n_dup_r += 1
    print(f"  {a}|{b}: {r}")
print(f"genuine target-level near-dups in random sample: {n_dup_r}/40")
