import os, sys, glob, time, shutil
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

torch.set_num_threads(8)

class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        f = 32
        self.e_conv1 = nn.Conv2d(3, f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(f*2, f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(f*2, f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(f*2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        rs = torch.split(x_r, 3, dim=1)
        for r in rs:
            x = x + r * (torch.pow(x, 2) - x)
        return x

SRC = "/hy-tmp/SLLFS/images"
DST = "/hy-tmp/SLLFS_zerodce/images"
SPLITS = ["train", "val", "test"]
BATCH = 16

dev = torch.device("cuda")
net = enhance_net_nopool()
sd = torch.load("/hy-tmp/zerodce_work/Epoch99.pth", map_location="cpu", weights_only=True)
if hasattr(sd, "state_dict"): sd = sd.state_dict()
net.load_state_dict(sd, strict=True)
net.eval().to(dev)
print("params:", sum(p.numel() for p in net.parameters()), flush=True)

pool = ThreadPoolExecutor(max_workers=8)

def load_img(p):
    img = Image.open(p).convert("RGB")
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a

def save_img(args):
    out_path, arr = args
    im = Image.fromarray((arr.clip(0, 1) * 255).round().astype(np.uint8))
    im.save(out_path, quality=95)

t_start = time.time()
total_done = 0
for split in SPLITS:
    files = sorted(glob.glob(os.path.join(SRC, split, "*.jpg")) + glob.glob(os.path.join(SRC, split, "*.png")))
    os.makedirs(os.path.join(DST, split), exist_ok=True)
    todo = [p for p in files if not os.path.exists(os.path.join(DST, split, os.path.basename(p)))]
    print(f"[{split}] total={len(files)} todo={len(todo)}", flush=True)
    buf, buf_sizes = [], []
    for p in todo:
        out_p = os.path.join(DST, split, os.path.basename(p))
        a = load_img(p)
        h, w = a.shape[:2]
        # flush if size changes
        if buf and buf_sizes[0] != (h, w):
            pass
        else:
            buf.append((out_p, a)); buf_sizes.append((h, w))
            if len(buf) < BATCH:
                continue
        if buf:
            arrs = np.stack([b[1] for b in buf])
            x = torch.from_numpy(arrs).permute(0, 3, 1, 2).contiguous().to(dev)
            with torch.no_grad():
                y = net(x)
            y = y.permute(0, 2, 3, 1).cpu().numpy()
            list(pool.map(save_img, [(b[0], y[i]) for i, b in enumerate(buf)]))
            total_done += len(buf)
            buf, buf_sizes = [], []
            if total_done % 200 < BATCH:
                rate = total_done / (time.time() - t_start)
                print(f"done={total_done} rate={rate:.1f} img/s elapsed={time.time()-t_start:.0f}s "
                      f"gpu_mem_alloc={torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)
    if buf:
        arrs = np.stack([b[1] for b in buf])
        x = torch.from_numpy(arrs).permute(0, 3, 1, 2).contiguous().to(dev)
        with torch.no_grad():
            y = net(x)
        y = y.permute(0, 2, 3, 1).cpu().numpy()
        list(pool.map(save_img, [(b[0], y[i]) for i, b in enumerate(buf)]))
        total_done += len(buf)
print(f"ALL DONE total={total_done} elapsed={(time.time()-t_start)/60:.1f}min", flush=True)
