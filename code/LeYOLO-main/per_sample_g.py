import cv2, numpy as np, torch, os
from ultralytics import YOLO
WEIGHTS = "/hy-tmp/LeYOLO-main/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"
IMGS = ["img000517", "img001547", "img006167", "img008605", "img018720", "img031139", "img061110"]
model = YOLO(WEIGHTS); net = model.model.cuda().eval()
G_maps, order = {}, []
def gh(name):
    def h(m, i, o):
        if name not in G_maps: order.append(name)
        G_maps[name] = o.detach().cpu()[0, 0]
    return h
for name, m in net.named_modules():
    if m.__class__.__name__ == "EdgeFcaV2":
        m.spatial_gate.register_forward_hook(gh(name))
print("per-sample G (box_mean / bg_mean):")
for img_id in IMGS:
    img = cv2.imread("/hy-tmp/SLLFS/images/test/%s.jpg" % img_id)
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
    x = torch.from_numpy(x / 255.0).float().permute(2, 0, 1)[None].cuda()
    with torch.no_grad(): _ = net(x)
    mask = np.zeros((640, 640), dtype=np.uint8)
    txt = "/hy-tmp/SLLFS/labels/test/%s.txt" % img_id
    if os.path.exists(txt):
        for line in open(txt):
            p = line.split()
            if len(p) != 5: continue
            _, cx, cy, w, h = map(float, p)
            x0, x1 = int((cx - w/2) * 640), int((cx + w/2) * 640)
            y0, y1 = int((cy - h/2) * 640), int((cy + h/2) * 640)
            mask[max(y0,0):max(y1,0), max(x0,0):max(x1,0)] = 1
    parts = []
    for name in order:
        G = G_maps[name].numpy(); n = G.shape[-1]
        tag = "P5" if n == 20 else "P4"
        m_n = cv2.resize(mask, (n, n), interpolation=cv2.INTER_AREA) > 0
        if m_n.any():
            parts.append("%s %.2f/%.2f" % (tag, G[m_n].mean(), G[~m_n].mean()))
        else:
            parts.append("%s n/a" % tag)
    print(" ", img_id, " | ".join(parts))
