import sys, os, cv2
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(3)

sys.path.insert(0, "/hy-tmp/LeYOLO-main")
from ultralytics import YOLO

CKPTS = [
    ("seed42-tinyS(s变体)", "/hy-tmp/LeYOLO-main/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"),
    ("seed5-tyM(s变体)",    "/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-m/weights/best.pt"),
    ("seed42-tinyN(n变体)", "/hy-tmp/LeYOLO-main/runs/detect/Tinyolo_s_topkedge21/weights/tiny_s_slli.pt"),
    ("seed5-tyS(n变体)",    "/hy-tmp/LeYOLO-main/runs/detect/5seed-ty-s/weights/best.pt"),
]
SAMPLES = ["img037004", "img029965", "img108101", "img023644", "img038566", "img023316", "img039160", "img039853"]

def boxes_of(stem):
    bs = []
    for line in open(f"/hy-tmp/SLLFS/labels/test/{stem}.txt"):
        p = line.split()
        if len(p) == 5:
            _, cx, cy, w, h = map(float, p)
            bs.append(((cx-w/2)*640, (cy-h/2)*640, (cx+w/2)*640, (cy+h/2)*640))
    return bs

def stats_for(g, boxes):
    bm = np.zeros((640, 640), bool)
    for x0, y0, x1, y1 in boxes:
        bm[max(int(y0)-10, 0):min(int(y1)+10, 640), max(int(x0)-10, 0):min(int(x1)+10, 640)] = True
    bg = np.zeros((640, 640), bool); bg[40:600, 40:600] = True; bg &= ~bm
    return g[bm].mean() - g[bg].mean(), g[bm].mean(), g[bg].mean(), g.mean(), g.std()

print(f"{'checkpoint':22s} {'node':9s} {'contrast':>9s} {'G_box':>7s} {'G_bg':>7s} {'G_mean':>7s} {'G_std':>7s}")
for tag, w in CKPTS:
    model = YOLO(w)
    net = model.model.cpu().eval()
    Gm = {}
    def gh(name):
        def h(m, i, o): Gm[name] = o.detach().cpu()
        return h
    names = []
    for name, m in net.named_modules():
        if m.__class__.__name__ == "EdgeFcaV2":
            m.spatial_gate.register_forward_hook(gh(name)); names.append(name)
    acc = {n: [] for n in names}
    for stem in SAMPLES:
        img = cv2.imread(f"/hy-tmp/SLLFS/images/test/{stem}.jpg")
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
        t = torch.from_numpy(x/255.0).float().permute(2, 0, 1)[None]
        Gm.clear()
        with torch.no_grad(): net(t)
        bx = boxes_of(stem)
        for n in names:
            g = F.interpolate(Gm[n], size=(640, 640), mode="bilinear")[0, 0].numpy()
            acc[n].append(stats_for(g, bx))
    for n in names:
        a = np.array(acc[n])
        tag2 = "P5" if "model.19" in n else ("P4td" if "model.22" in n else "P4bu")
        print(f"{tag:22s} {tag2:9s} {a[:,0].mean():+9.3f} {a[:,1].mean():7.3f} {a[:,2].mean():7.3f} {a[:,3].mean():7.3f} {a[:,4].mean():7.3f}")
    del model, net
    print()
