import sys, os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---- official Zero-DCE DCE-Net (79,416 params) ----
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image_1, enhance_image, x_r

def load_model(weights_path):
    net = enhance_net_nopool()
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if hasattr(sd, "state_dict"): sd = sd.state_dict()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net

def enhance_pil(net, img_pil):
    x = torch.from_numpy(np.asarray(img_pil.convert("RGB"), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        _, enhanced, _ = net(x)
    enhanced = enhanced.clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((enhanced * 255).round().astype(np.uint8))

if __name__ == "__main__":
    net = load_model("Epoch99.pth")
    print("params:", sum(p.numel() for p in net.parameters()))
    img_root = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/SLLFS/images/train"
    out_root = sys.argv[2] if len(sys.argv) > 2 else "/hy-tmp/zerodce_work/samples"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out_root, exist_ok=True)
    # deterministic diverse pick: evenly spaced over the sorted list
    files = sorted(glob.glob(os.path.join(img_root, "*.jpg")) + glob.glob(os.path.join(img_root, "*.png")))
    print("total imgs:", len(files))
    idxs = np.linspace(0, len(files) - 1, n).round().astype(int).tolist()
    for i in idxs:
        p = files[i]
        t0 = time.time()
        img = Image.open(p)
        enh = enhance_pil(net, img)
        dt = time.time() - t0
        a_in = np.asarray(img.convert("L"), dtype=np.float32)
        a_en = np.asarray(enh.convert("L"), dtype=np.float32)
        name = os.path.splitext(os.path.basename(p))[0]
        # side-by-side comparison
        W = img.width + enh.width + 8
        H = max(img.height, enh.height)
        canvas = Image.new("RGB", (W, H), (255, 0, 0))
        canvas.paste(img.convert("RGB"), (0, 0))
        canvas.paste(enh, (img.width + 8, 0))
        cmp_path = os.path.join(out_root, f"cmp_{name}.jpg")
        canvas.save(cmp_path, quality=92)
        print(f"{os.path.basename(p)} size={img.size} lum_mean {a_in.mean():.1f}->{a_en.mean():.1f} "
              f"lum_p95 {np.percentile(a_in,95):.1f}->{np.percentile(a_en,95):.1f} ({dt:.2f}s, cpu) -> {cmp_path}")
    print("DONE")
