import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class RetinexDecom(nn.Module):
    def __init__(self,hidden_dim=18):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(hidden_dim*3, out_channels=4,kernel_size=1,padding=0,stride=1)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
    def forward(self,x):
        x = self.layer1(x)
        x1 = self.layer2(x) + x
        x2 = self.layer3(x) + x
        x3 = self.layer4(x) + x
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.conv1x1(x)
        R, L = x[:,:3,:,:], x[:, -1:, :, :]
        R = self.tanh1(R)
        L = self.tanh2(L)
        return R, L

class ECAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        y=self.gap(x)
        y=y.squeeze(-1).permute(0,2,1)
        y=self.conv(y)
        y=self.sigmoid(y)
        y=y.permute(0,2,1).unsqueeze(-1)
        return x*y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.atten = ECAttention()
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class IlluminationEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(in_channels=1,out_channels=8)
        self.down1 = Down(in_channels=8,out_channels=8)
        self.down2 = Down(in_channels=16,out_channels=8)
        self.maxpool1 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels=8,out_channels=8))
        self.maxpool2 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels=16,out_channels=8))
        self.ca1 = ECAttention()
        self.ca2 = ECAttention()
        self.ca3 = ECAttention()
        self.ca4 = ECAttention()
        self.outc = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1,stride=1)
    def forward(self,x):
        x = self.inc(x)
        x1 = self.down1(x)
        x1 = self.ca1(x1) + x1
        x2 = self.maxpool1(x)
        x2 = self.ca2(x2) + x2
        x = torch.cat((x1,x2),dim=1)
        x3 = self.down2(x)
        x3 = self.ca3(x3) + x3
        x4 = self.maxpool2(x)
        x4 = self.ca4(x4) + x4
        x = torch.cat((x3,x4),dim=1)
        return self.outc(x)

class colorMatrix(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels=in_channels,out_channels=mid_channels)
        self.eca1 = ECAttention()
        self.layer1 = nn.Sequential(
            self.dwconv(in_dim=mid_channels,out_dim=mid_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AvgPool2d(2)
        )
        self.eca2 = ECAttention()
        self.layer2 = nn.Sequential(
            self.dwconv(in_dim=mid_channels,out_dim=mid_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AvgPool2d(2)
        )
        self.eca3 = ECAttention()
        self.layer3 = nn.Sequential(
            self.dwconv(in_dim=mid_channels,out_dim=mid_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2)
        )
        self.eca4 = ECAttention()
        self.outc = DoubleConv(in_channels=mid_channels,out_channels=out_channels)
    @staticmethod
    def dwconv(in_dim,out_dim):
        return nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=1,padding=0,stride=1,groups=in_dim)
    def forward(self,x):
        x = self.inc(x)
        x = self.eca1(x)
        x = self.layer1(x)
        x = self.eca2(x)
        x = self.layer2(x)
        x = self.eca3(x)
        x = self.layer3(x)
        x = self.eca4(x)
        x = self.outc(x)
        return x

class ReflectanceCCM(nn.Module):
    def __init__(self, in_dim=3, out_dim=12):
        super().__init__()
        self.isp_body = colorMatrix(in_channels=in_dim, out_channels=out_dim,mid_channels=out_dim*2)
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim * 2 * 2, hidden_features=out_dim, out_features=9)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccm_base = nn.Parameter(torch.ones(3,3), requires_grad=True)
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[1], [0]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    def forward(self, img):
        out = self.isp_body(img)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = img.shape[0]
        ccm = torch.reshape(out, shape=(b, 3, 3)) + self.ccm_base
        img = img.permute(0, 2, 3, 1)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccm[i, :, :]) for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)
        return img

class GammaEnhance(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        self.ien = IlluminationEnhanceNet()
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim * 2 * 2, hidden_features=out_dim, out_features=2)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccw_base = nn.Parameter(torch.eye((1)), requires_grad=True)
        self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)
    def apply_color(self, image, ccw):
        shape = image.shape
        image = image.view(-1, 1)
        image = torch.tensordot(image, ccw, dims=[[1], [0]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)
    def forward(self, L):
        out = self.ien(L)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = L.shape[0]
        gamma = out[:, 0:1] + self.gamma_base
        ccw = out[:,1:] + self.ccw_base
        img = L.permute(0, 2, 3, 1)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccw[i, :]) ** gamma[i, :] for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)
        return img

# ----------------------
# 【关键修改】适配 Ultralytics
# ----------------------
class EMVEnhance(nn.Module):
    """Ultralytics 版本的 EMV 增强模块"""
    def __init__(self, **kwargs):
        super().__init__()
        self.rdn = RetinexDecom()
        self.ccm = ReflectanceCCM()
        self.gamma = GammaEnhance()

    def forward(self, x):
        # 完全和原文一致
        r, l = self.rdn(x)
        r_high = self.ccm(r)
        l_high = self.gamma(l)
        img_high = r_high * l_high

        return img_high
