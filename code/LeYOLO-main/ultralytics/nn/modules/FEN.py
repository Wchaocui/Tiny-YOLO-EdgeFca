import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from ultralytics.nn.modules.conv import DWConv
from ultralytics.nn.modules.block import GhostBottleneck as SpaBlock
class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0, bias=False)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0,bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0,bias=False))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0,bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0,bias=False))

    def forward(self, x):
        _, _, H, W = x.shape
        # print(x.dtype)
        x = self.fpre(x)
        # print(x.dtype)
        x_freq = torch.fft.rfft2(x.float(), norm='backward')
        # x_freq.shape = B, C, H, W//2+1
        mag = torch.abs(x_freq)#
        # print(mag.dtype)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        # print(mag.dtype)
        pha = self.process2(pha)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        # imag = x_freq.imag

        x_out = torch.complex(real.float(), imag.float())
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward').float()

        return (x_out + x)
class FPN(nn.Module):
    def __init__(self,c_in,c):
        super(FPN, self).__init__()
        self.pconv = nn.Conv2d(c_in, c, 1, 1,0,bias=False)
        self.fblock1 = FreBlock(c)
        self.fblock2 = FreBlock(c)
        self.fblock3 = FreBlock(c*2)
        # self.fblock3 = nn.Sequential(FreBlock(c*2),
        #                              nn.Conv2d(c*2,c,1,1,0,bias=False))
        # self.fblock5 = nn.Sequential(FreBlock(c*2),
        #                              nn.Conv2d(c*2,c,1,1,0,bias=False))
        # self.outconv = nn.Sequential(FreBlock(c*2),
        #                              nn.Conv2d(c*2,c,1,1,0,bias=False))
        self.outconv = nn.Conv2d(c*2, c, 1, 1, 0,bias=False)
    def forward(self, x):
        x = self.pconv(x)
        # print(x.dtype)
        x1 = self.fblock1(x)
        x2 = self.fblock2(x1)
        # x2 = self.fblock3(x1)
        # x4 = self.fblock4(torch.cat([x3, x2], dim=1))
        # x5 = self.fblock5(torch.cat([x4, x1], dim=1))
        x = torch.cat([x.float(),x2], dim=1)
        # print(x.dtype)
        x = self.fblock3(x)
        out = self.outconv(x)
        return out

class SpaBlock1(nn.Module):
    def __init__(self,c1,c2,k,s):
        super(SpaBlock1, self).__init__()
        self.pconv1 = nn.Conv2d(c1,c1//2,1,bias=False)
        self.conv = nn.Sequential(nn.Conv2d(c1//2,c1//2,k,s,bias=False,groups=c1//2,padding=1),nn.BatchNorm2d(c1//2),
                                   nn.Conv2d(c1//2,c1//2,1,bias=False),
                                   nn.SiLU(inplace=True))
        self.pconv2 = nn.Conv2d(c1,c2,1,bias=False)

    def forward(self, x):
        x = self.pconv1(x)
        # print(x.shape)
        x1 = self.conv(x)
        x2 = self.conv(x1)
        x3 = self.conv(x2)
        # print(x3.shape)
        x_1 = self.pconv2(torch.cat((x1,x3),1))
        return x_1


class FEN(nn.Module):
    def __init__(self,c1,c2,k,s=1):
        super(FEN, self).__init__()
        self.gconv = SpaBlock1(c1, c2, k, s)
        self.fpn = FPN(c2,c2)

    def forward(self, x):
        x = self.gconv(x)
        x2 = self.fpn(x)
        return (x+x2)

if __name__ == '__main__':
    input = torch.randn(2, 32, 64, 64)
    model = FEN(32,96,3,1)
    out = model(input)
    print(out.shape)