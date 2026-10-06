import os, glob, json, time
import numpy as np
from PIL import Image
from multiprocessing import Pool

ROOT = "/hy-tmp/SLLFS/images"
SPLITS = ["train", "val", "test"]

def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size+1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def load(p):
    try:
        return dhash(p)
    except Exception:
        return None

if __name__ == "__main__":
    t0 = time.time()
    files, splits = [], []
    for s in SPLITS:
        fs = sorted(glob.glob(os.path.join(ROOT, s, "*.jpg")) + glob.glob(os.path.join(ROOT, s, "*.png")))
        files += fs; splits += [s] * len(fs)
    print(f"total {len(files)} images", flush=True)
    with Pool(4) as pool:
        hashes = pool.map(load, files, chunksize=64)
    keep = [i for i, h in enumerate(hashes) if h is not None]
    H = np.packbits(np.stack([hashes[i] for i in keep]), axis=1).astype(np.uint64)[:, 0]
    sp = np.array([splits[i] for i in keep])
    names = [os.path.basename(files[i]) for i in keep]
    print(f"hashed {len(H)} images in {time.time()-t0:.0f}s", flush=True)

    # popcount lookup
    pc = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        pc[i] = bin(i).count("1")
    def popcount(x):
        return pc[(x.view(np.uint8)).reshape(len(x), -1)].sum(axis=1)

    N = len(H)
    cross_pairs, within_pairs = [], []
    mins = []
    CH = 2048
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        block = H[i0:i1, None] ^ H[None, :]
        d = np.zeros((i1 - i0, N), dtype=np.uint8)
        b = block.view(np.uint8).reshape(i1 - i0, N, 8)
        for k in range(8):
            d += pc[b[:, :, k]]
        # only upper triangle
        col = np.arange(N)[None, :]
        row = np.arange(i0, i1)[:, None]
        mask = col > row
        d = np.where(mask, d, 99)
        mins.append(d.min(axis=1))
        rr, cc = np.nonzero(d <= 6)
        for r, c in zip(rr, cc):
            a, bidx = i0 + r, c
            pair = (names[a], names[bidx], int(d[r, c]))
            if sp[a] != sp[bidx]:
                cross_pairs.append(pair + (sp[a], sp[bidx]))
            else:
                within_pairs.append(pair)
    mins = np.concatenate(mins)
    print(f"scan done in {time.time()-t0:.0f}s total", flush=True)
    print(f"neighbor-distance stats: min={mins.min()} p1={np.percentile(mins,1):.0f} median={np.median(mins):.0f}", flush=True)
    print(f"near-dup pairs (hamming<=6/64): cross-split={len(cross_pairs)}, within-split={len(within_pairs)}", flush=True)
    for p in cross_pairs[:10]: print("CROSS", p, flush=True)
    for p in within_pairs[:10]: print("WITHIN", p, flush=True)
    json.dump({"cross": cross_pairs, "within": within_pairs[:200]}, open("dup_scan_result.json", "w"), indent=1)
    print("saved dup_scan_result.json", flush=True)
