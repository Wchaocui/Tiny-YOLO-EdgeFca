import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from ultralytics import YOLO

# 1.加载模型提取P5
model = YOLO("../runs/detect/Tinyolo_s_topkedge21/weights/best.pt")
net = model.model

feat_save = None
def hook_func(module, inp, out):
    global feat_save
    feat_save = out.detach().cpu()

target_layer = net.model[15]
handle = target_layer.register_forward_hook(hook_func)

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_path = "/picture/img2.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError("图片路径错误，无法读取！")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
input_tensor = transform(img_rgb).unsqueeze(0)

with torch.no_grad():
    _ = net(input_tensor)
feature_p5 = feat_save
handle.remove()

# 2.池化算子
max5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
avg3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

def sppf_forward(x):
    y1 = x
    y2 = max5(y1)
    y3 = max5(y2)
    y4 = max5(y3)
    return y3, y4

def mma_forward(x):
    y1 = x
    y2 = max5(y1)
    y3 = max5(y2)
    y4 = avg3(y3)
    return y3, y4

y3, y4_sppf = sppf_forward(feature_p5)
_, y4_mma = mma_forward(feature_p5)

# 转为可视化numpy
def to_np(t):
    return t.mean(dim=1).squeeze().cpu().numpy()

f_y3 = to_np(y3)
f_y4_sppf = to_np(y4_sppf)
f_y4_mma = to_np(y4_mma)

feat_list = [f_y3, f_y4_sppf, f_y4_mma]
title_list = [
    "$\mathbf{Y}^{(3)}$ (after two successive $5\\times5$ max-pooling)",
    "SPPF $\mathbf{Y}^{(4)}$ (final $5\\times5$ max-pooling)",
    "MMA-SPPF $\mathbf{Y}^{(4)}$ (final $3\\times3$ avg-pooling)"
]

# 绘图1行3列
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
fig, axes = plt.subplots(1, 3, figsize=(12,4.8), dpi=300)
vmin = np.min(feat_list)
vmax = np.max(feat_list)

for idx, ax in enumerate(axes):
    ax.imshow(feat_list[idx], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title_list[idx], pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("Y3_Y4_compare.png", bbox_inches="tight", dpi=300)
plt.show()
