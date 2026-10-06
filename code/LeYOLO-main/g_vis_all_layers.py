import cv2, numpy as np, torch, os, sys, glob
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/hy-tmp/LeYOLO-main")
from ultralytics import YOLO

torch.set_num_threads(4)
WEIGHTS = "/hy-tmp/LeYOLO-main/runs/detect/tiny_m_topkedge21/weights/tiny_m.pt"
OUT = "/hy-tmp/LeYOLO-main/gating_vis_all"
os.makedirs(OUT, exist_ok=True)

# ---- pick samples: one SLLI low-light test image + one well-exposed SPARK image outside SLLI ----
SAMPLES = [
    ("/hy-tmp/SLLFS/images/test/img039853.jpg",
     "/hy-tmp/SLLFS/labels/test/img039853.txt", "SLLI low-light sample", False),
    ("/hy-tmp/normal_samples/img066481.jpg",
     "/hy-tmp/normal_samples/img066481.txt", "Well-exposed sample (outside SLLI)", False),
]
for _p, _l, _t, _o in SAMPLES: print(f"{_t}: {_p}")

# ---- BWR colormap via matplotlib ----
import matplotlib
cmap = matplotlib.colormaps["bwr"]
BWR = (cmap(np.linspace(0, 1, 256))[:, :3][:, ::-1] * 255).astype(np.uint8)  # BGR for cv2
def bwr_color(a):
    idx = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return BWR[idx]

# ---- load model & hooks ----
model = YOLO(WEIGHTS)
net = model.model.cuda().eval()

G_maps, E_maps, module_names = {}, {}, []
def mkG(name):
    def h(m, i, o): G_maps[name] = o.detach().cpu()
    return h
def mkE(name):
    def h(m, i, o):
        # spatial_gate input = post-ReLU edge features; channel-mean = E
        E_maps[name] = i[0].detach().mean(1, keepdim=True).cpu()
    return h

for name, m in net.named_modules():
    if m.__class__.__name__ == "EdgeFcaV2":
        m.spatial_gate.register_forward_hook(mkG(name))
        m.spatial_gate.register_forward_hook(mkE(name))
        module_names.append(name)

# map module index → label (by spatial resolution or model order)
# model.19 = P5 backbone output, model.22 = P4 top-down, model.33 = P4 bottom-up
LAYER_LABELS = {"model.19": "P5 (backbone output)",
                "model.22": "P4 (top-down fusion)",
                "model.33": "P4 (bottom-up fusion)"}
print("hooked:", module_names)

def boxes_of(lp):
    boxes = []
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) == 5:
                _, cx, cy, w, h = map(float, p)
                boxes.append((int((cx-w/2)*640), int((cy-h/2)*640),
                              int((cx+w/2)*640), int((cy+h/2)*640)))
    return boxes

def up640(t):
    return F.interpolate(t, size=(640,640), mode="bilinear")[0,0].numpy()

PANEL = 380  # panel size in pixels
BAR_H = 36   # label bar height
FONT = cv2.FONT_HERSHEY_SIMPLEX

def label_bar(text, color=(255,255,255), bg=40):
    bar = np.full((BAR_H, PANEL, 3), bg, dtype=np.uint8)
    # split text if too wide
    font_scale = 0.62
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    if tw > PANEL - 20:
        font_scale = 0.48
        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x = max(8, (PANEL - tw) // 2)
    cv2.putText(bar, text, (x, 26), FONT, font_scale, color, thickness, cv2.LINE_AA)
    return bar

def make_heat_panel(arr, img, boxes, title, overlay=True):
    heat = bwr_color(arr)
    ov = cv2.addWeighted(img, 0.4, heat, 0.6, 0) if overlay else heat
    for b in boxes:
        cv2.rectangle(ov, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
    ov = cv2.resize(ov, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    return np.vstack([label_bar(title), ov])

def make_input_panel(img, boxes, title):
    inp = img.copy()
    for b in boxes:
        cv2.rectangle(inp, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
    inp = cv2.resize(inp, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    return np.vstack([label_bar(title, color=(0,255,0)), inp])

# ---- forward and build figure ----
rows_fig = []
sep_v = np.full((BAR_H + PANEL, 6, 3), 220, dtype=np.uint8)

for row_idx, (img_path, lbl_path, row_title, ov_flag) in enumerate(SAMPLES):
    img = cv2.resize(cv2.imread(img_path), (640, 640))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
    G_maps.clear(); E_maps.clear()
    with torch.no_grad():
        _ = net(x)
    boxes = boxes_of(lbl_path)
    panels = [make_input_panel(img, boxes, row_title)]

    # order: P4td, P4bu, P5 (as user requested)
    display_order = ["model.22", "model.33", "model.19"]
    for mod_name in display_order:
        if mod_name not in G_maps:
            continue
        layer = LAYER_LABELS.get(mod_name, mod_name)
        # E: normalize per-map
        e = up640(E_maps[mod_name])
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # G: already in [0,1]
        g = up640(G_maps[mod_name])
        panels.append(make_heat_panel(e, img, boxes, f"E | {layer}", ov_flag))
        panels.append(make_heat_panel(g, img, boxes, f"G | {layer}", ov_flag))

    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, sep_v, p])
    rows_fig.append(row)
    Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)).save(
        os.path.join(OUT, f"row{row_idx+1}.jpg"), quality=95)
    print(f"row '{row_title}': {len(panels)} panels")

sep_h = np.full((8, rows_fig[0].shape[1], 3), 220, dtype=np.uint8)
fig = np.vstack([rows_fig[0], sep_h, rows_fig[1]])

os.makedirs(OUT, exist_ok=True)
out_pdf = os.path.join(OUT, "gating_EG_all_layers.pdf")
out_jpg = os.path.join(OUT, "gating_EG_all_layers.jpg")
Image.fromarray(cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)).save(out_pdf, "PDF", resolution=150)
Image.fromarray(cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)).save(out_jpg, quality=95)
print(f"\nsaved: {out_pdf}")
print(f"saved: {out_jpg}")
print(f"figure size: {fig.shape[1]}x{fig.shape[0]}")

# ---- quantitative stats ----
print("\n===== Quantitative stats =====")
for img_path, lbl_path, title, _ov in SAMPLES:
    img = cv2.resize(cv2.imread(img_path), (640, 640))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
    G_maps.clear(); E_maps.clear()
    with torch.no_grad():
        _ = net(x)
    boxes = boxes_of(lbl_path)
    mask = np.zeros((640, 640), dtype=bool)
    for b in boxes: mask[b[1]:b[3], b[0]:b[2]] = True
    bg = ~mask

    for mod_name in display_order:
        if mod_name not in G_maps: continue
        layer = LAYER_LABELS.get(mod_name, mod_name)
        g = up640(G_maps[mod_name])
        e = up640(E_maps[mod_name])
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
        print(f"{title:22s} {layer:25s} E: box={e_norm[mask].mean():.3f} bg={e_norm[bg].mean():.3f}  "
              f"G: box={g[mask].mean():.3f} bg={g[bg].mean():.3f}  contrast={g[mask].mean()-g[bg].mean():+.3f}")
