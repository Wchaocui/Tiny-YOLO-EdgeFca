import os, glob, time
import numpy as np
from PIL import Image

IMG, LBL = "/hy-tmp/SLLFS/images", "/hy-tmp/SLLFS/labels"
data = {}
for s in ["train", "val", "test"]:
    for lp in glob.glob(os.path.join(LBL, s, "*.txt")):
        stem = os.path.basename(lp)[:-4]
        bs = []
        for line in open(lp):
            p = line.split()
            if len(p) >= 5:
                bs.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if bs:
            data[(s, stem)] = bs

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2
    bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    ix = max(0, min(ax2,bx2)-max(ax1,bx1)); iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy; ua = a[3]*a[4]+b[3]*b[4]-inter
    return inter/ua if ua > 0 else 0.0

splits = ["train", "val", "test"]
t0 = time.time()
cands = []
for si in range(3):
    for sj in range(si+1, 3):
        A = [(k, b) for k, bs in data.items() for b in bs if k[0] == splits[si]]
        B = {splits[sj]: [ (k, b) for k, bs in data.items() for b in bs if k[0] == splits[sj] ]}
        # group B by class for speed
        from collections import defaultdict
        Bc = defaultdict(list)
        for k, b in B[splits[sj]]:
            Bc[b[0]].append((k, b))
        for k, b in A:
            for kb, bb in Bc.get(b[0], []):
                io = iou(b, bb)
                if io >= 0.5:
                    cands.append((k, b, kb, bb, io))
print(f"label-level candidates (same class, box IoU>=0.5, cross-split): {len(cands)}  [{time.time()-t0:.0f}s]", flush=True)

# crop-level verification for candidates
def img_path(k):
    s, stem = k
    for ext in (".jpg", ".png"):
        p = os.path.join(IMG, s, stem + ext)
        if os.path.exists(p): return p
    return None

def crop_norm(path, box, size=96, pad=0.15):
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    H, W = img.shape
    x1 = max(0, int((box[1]-box[3]/2-pad)*W)); y1 = max(0, int((box[2]-box[4]/2-pad)*H))
    x2 = min(W, int((box[1]+box[3]/2+pad)*W)); y2 = min(H, int((box[2]+box[4]/2+pad)*H))
    c = img[y1:y2, x1:x2]
    if c.size == 0: return None
    c = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((size, size)), dtype=np.float32)
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    return (c - lo) / max(hi - lo, 1e-3)

real_dups = []
for k, b, kb, bb, io in cands:
    pa, pb = img_path(k), img_path(kb)
    if not pa or not pb: continue
    ca, cb = crop_norm(pa, b), crop_norm(pb, bb)
    if ca is None or cb is None: continue
    mad = float(np.abs(ca - cb).mean()) * 255
    if mad < 15:
        real_dups.append((k[1], kb[1], round(io, 3), round(mad, 1), k[0], kb[0]))
print(f"verified target-level near-duplicates (crop MAD<15/255): {len(real_dups)}")
for r in real_dups[:20]: print("  DUP", r)
print(f"total time {time.time()-t0:.0f}s")
