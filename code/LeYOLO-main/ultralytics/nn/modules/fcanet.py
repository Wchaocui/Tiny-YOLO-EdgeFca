import torch
import torch.nn as nn
import math
from torchvision.models import ResNet
import torch.nn.functional as F


all=('conv3x3',
     'FcaBottleneck','FcaBasicBlock',
     'fcanet34','fcanet50','fcanet101','fcanet152',
     'MultiSpectralAttentionLayer','MultiSpectralDCTLayer',
     'EdgeFcaLayer',"EMSSHead",'EnhancedEdgeFcaLayer')

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FcaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        global _mapper_x, _mapper_y
        super(FcaBottleneck, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fcanet34(num_classes=11, pretrained=False):
    """Constructs a FcaNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet50(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet101(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet152(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = 16
        self.dct_h = dct_h
        self.dct_w = self.dct_h

        mapper_x, mapper_y = get_freq_indices('top16')
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (self.dct_w // 8) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, self.dct_w , mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // self.reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x

        if h != self.dct_h or w != self.dct_w:
            x_pooled =  torch.nn.functional.adaptive_avg_pool2d(x,(self.dct_h,self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        # print(self.dct_h,self.dct_w,self.reduction)
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = (
                            self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y))

        return dct_filter




import pywt  # 需要安装PyWavelets

# 动态残差融合模块
class DynamicFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 可学习权重系数，每个通道独立
        self.alpha = nn.Parameter(torch.zeros(1, channel, 1, 1))  # 形状为 [1, C, 1, 1]

    def forward(self, res, output):
        # 动态调节融合比例，逐通道加权
        return res + torch.sigmoid(self.alpha) * output
class EdgeFcaLayer(nn.Module):
    def __init__(self, channel, dct_h, wavelet_type='db4'):
        super().__init__()
        self.supported_wavelets = ['haar', 'db4', 'sym8']  # 支持的小波类型
        assert wavelet_type in self.supported_wavelets, f"Unsupported wavelet: {wavelet_type}"
        self.wavelet = wavelet_type
        # 边缘增强卷积
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),  # 深度可分离卷积
            nn.GELU(),  # 使用GELU激活函数
            nn.Conv2d(channel, channel, 1, bias=False),  # 逐点卷积
            nn.Sigmoid()  # 输出注意力图
        )
        # 频域注意力
        self.fca = MultiSpectralAttentionLayer(channel,dct_h)  # 前述频域注意力模块
        # 动态残差融合
        self.fusion = DynamicFusion(channel)

    def wavelet_transform(self, x):
        """执行小波变换，支持多种小波基"""
        x_np = x.data.float().cpu().numpy()
        edge_maps = []

        for i in range(x.shape[1]):
            try:
                # 根据指定小波类型执行变换
                coeffs = pywt.dwt2(x_np[:, i, :, :], self.wavelet)
                cA, (cH, cV, cD) = coeffs

                # 边缘图计算方式可根据小波特性调整
                if self.wavelet == 'haar':
                    # Haar小波适合提取清晰边缘
                    edge_map = torch.from_numpy(cH + cV).to(x.device)
                elif self.wavelet == 'db4':
                    # Db4适合平滑边缘，增强低频细节
                    edge_map = torch.from_numpy(cH + cV + 0.5 * cD).to(x.device)
                elif self.wavelet == 'sym8':
                    # Sym8适合保留纹理细节
                    edge_map = torch.from_numpy(cH + cV + 0.3 * cD).to(x.device)

                edge_maps.append(edge_map)
            except Exception as e:
                print(f"Wavelet transform error for {self.wavelet}: {e}")
                # 出错时返回原始通道
                edge_maps.append(torch.zeros_like(x[:, i:i + 1, :, :]))

        # 恢复通道维度
        edge_map = torch.stack(edge_maps, dim=1)
        return edge_map

    def forward(self, x):
        # 小波分解提取边缘
        with torch.no_grad():
            edge_map = self.wavelet_transform(x)

        # 上采样到原始尺寸
        edge_map = F.interpolate(edge_map, size=x.shape[2:], mode='bicubic', align_corners=False)

        # 保持与输入相同的数据类型
        if x.dtype == torch.float16:
            edge_map = edge_map.half()

        # 边缘增强
        edge_enhanced = self.edge_conv(x * edge_map)

        # 频域注意力加权
        output = self.fca(edge_enhanced)

        # 动态残差融合
        return self.fusion(x, output)


'''改进版edgeflayer'''
class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
#. 可学习的边缘-噪声分离模块
class EdgeNoiseSeparator(nn.Module):
    def __init__(self, in_channels, base_kernel_size=5):
        super().__init__()
        # 多尺度边缘检测
        self.edge_conv = nn.Sequential(
            # 方向敏感卷积核
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=base_kernel_size,
                      padding=base_kernel_size // 2, groups=in_channels, bias=False),
            nn.GELU(),
            # 空间连续性约束
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(in_channels)
        )

        # 噪声模式学习
        self.noise_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            SpatialAttention(kernel_size=7),  # 空间注意力聚焦离散点
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 边缘特征 (强调连续结构)
        edge_feat = self.edge_conv(x)

        # 噪声特征 (捕捉离散点)
        noise_mask = self.noise_net(x)

        # 分离增强
        return edge_feat * (1 - noise_mask), noise_mask
# 改进的频域注意力（抑制噪声频段）
import numpy as np


#3. 动态特征融合（带边缘置信度门控）
class EdgeAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_confidence = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_raw, x_edge, x_noise_mask):
        # 边缘置信度估计 (高置信度=连续边缘)
        confidence = self.edge_confidence(x_edge)

        # 噪声掩码反向加权
        fusion_weight = confidence * (1 - x_noise_mask)

        return x_raw * (1 - fusion_weight) + x_edge * fusion_weight


class EnhancedEdgeFcaLayer(nn.Module):
    def __init__(self, channels, dct_h=8):
        super().__init__()
        self.separator = EdgeNoiseSeparator(channels)
        self.freq_attn = MultiSpectralAttentionLayer(channels, dct_h)
        self.fusion = EdgeAwareFusion(channels)

    def forward(self, x):
        # 阶段1：边缘-噪声分离
        edge_feat, noise_mask = self.separator(x)

        # 阶段2：目标导向频域增强
        edge_enhanced = self.freq_attn(edge_feat)

        # 阶段3：置信度门控融合
        output = self.fusion(x, edge_enhanced, noise_mask)

        return output
if __name__ == '__main__':
    input = torch.randn(8,32,128,128)
    layer = MultiSpectralAttentionLayer(32,80)
    output = layer(input)

    print(output.shape)
