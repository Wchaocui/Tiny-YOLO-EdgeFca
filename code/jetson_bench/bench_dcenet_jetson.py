# =====================================================================
# Jetson Orin Nano 上测量 DCE-Net(Zero-DCE)延迟 —— 用于补全 Table II 的 FPS
# 使用方法:
#   1. 把本脚本和 Epoch99.pth(Zero-DCE 官方权重)拷到 Jetson;
#   2. python3 bench_dcenet_jetson.py
#   3. 得到 t_dce (ms);FPS = 1000 / (t_dce + 25.34)   # 25.34ms = YOLOv11-n 实测
# 协议与论文一致:10 次预热 + 每图 100 次前向,前后 torch.cuda.synchronize()
# =====================================================================
import time
import numpy as np
import torch
import torch.nn as nn


class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        f = 32
        self.e_conv1 = nn.Conv2d(3, f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(f, f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(f * 2, f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for r in torch.split(x_r, 3, dim=1):
            x = x + r * (torch.pow(x, 2) - x)
        return x


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = enhance_net_nopool()
    sd = torch.load("Epoch99.pth", map_location="cpu", weights_only=True)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    net.load_state_dict(sd, strict=True)
    net.eval().to(dev)
    print(f"DCE-Net params: {sum(p.numel() for p in net.parameters()):,} on {dev}")

    x = torch.rand(1, 3, 640, 640, device=dev)
    with torch.no_grad():
        for _ in range(10):
            _ = net(x)
    if dev == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(100):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = net(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    t = float(np.mean(times))
    print(f"DCE-Net latency @640x640: {t:.2f} ms  (std {np.std(times):.2f})")
    t_yolo = 25.34  # YOLOv11-n on Orin Nano (Table VII)
    print(f"Two-stage pipeline: {t + t_yolo:.2f} ms -> FPS = {1000 / (t + t_yolo):.2f}")


if __name__ == "__main__":
    main()
