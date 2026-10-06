import json, os, glob
import numpy as np
from PIL import Image

ROOT = "/hy-tmp/SLLFS/images"
idx = {}
for s in ["train", "val", "test"]:
    for f in glob.glob(os.path.join(ROOT, s, "*.jpg")):
        idx[os.path.basename(f)] = f

def thumb(path, size=128):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)

d = json.load(open("dup_scan_result.json"))
cross = d["cross"]
print("sampled verification of cross-split dHash-0 pairs (pixel-level MAD on 128x128):")
rng = np.random.default_rng(0)
sample = rng.choice(len(cross), size=min(30, len(cross)), replace=False)
mads = []
for i in sample:
    a, b, hdist, sa, sb = cross[i]
    A, B = thumb(idx[a]), thumb(idx[b])
    mad = float(np.abs(A - B).mean())
    mads.append(mad)
    print(f"  {a} ({sa}) vs {b} ({sb}): dHash={hdist}, pixelMAD128={mad:.2f}")
mads = np.array(mads)
print(f"MAD stats over sample: min={mads.min():.2f} median={np.median(mads):.2f} max={mads.max():.2f}")
# baseline: random cross pairs for comparison
names = list(idx)
print("baseline random pairs:")
bm = []
for _ in range(30):
    a, b = rng.choice(names, 2, replace=False)
    bm.append(float(np.abs(thumb(idx[a]) - thumb(idx[b])).mean()))
bm = np.array(bm)
print(f"  random MAD: min={bm.min():.2f} median={np.median(bm):.2f} max={bm.max():.2f}")
