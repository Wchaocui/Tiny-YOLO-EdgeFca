import torch
import torch.nn as nn
import torch.nn.functional as F



class SmoothnessHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)  # 生成光滑性掩码
        self.laplacian = nn.Conv2d(1, 1, 3, padding=1, bias=False)  # 拉普拉斯算子
        self.laplacian.weight.data = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)

    def forward(self, x):
        smooth_mask = torch.sigmoid(self.conv(x))  # 预测光滑区域概率
        laplacian = torch.abs(self.laplacian(smooth_mask))  # 计算二阶导响应
        return smooth_mask * (1 - laplacian)  # 抑制纹理区域








class MSS(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # 预测光滑性掩码的卷积
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # 多尺度拉普拉斯算子
        self.laplacian_scales = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(num_scales)
        ])

        # 初始化拉普拉斯算子权重为可学习参数
        for laplacian in self.laplacian_scales:
            laplacian.weight.data = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
            laplacian.weight.requires_grad = True  # 设置为可学习

        # 梯度计算卷积（Sobel算子）
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.sobel_y.weight.data = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # 预测光滑性掩码
        smooth_mask = torch.sigmoid(self.conv(x))

        # 多尺度拉普拉斯响应
        laplacian_responses = []
        for laplacian in self.laplacian_scales:
            response = torch.abs(laplacian(smooth_mask))
            laplacian_responses.append(response)

        # 融合多尺度拉普拉斯响应
        laplacian_fused = torch.stack(laplacian_responses, dim=1).mean(dim=1)  # 平均融合

        # 计算梯度信息
        gradient_x = torch.abs(self.sobel_x(smooth_mask))
        gradient_y = torch.abs(self.sobel_y(smooth_mask))
        gradient = gradient_x + gradient_y

        # 结合拉普拉斯响应和梯度信息
        smoothness_weight = smooth_mask * (1 - laplacian_fused) * (1 - gradient)

        return smoothness_weight

# EnhancedMultiScaleSmoothnessHead

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))
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

class EMSS(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels

        # 轻量化掩码生成
        self.smooth_conv = nn.Sequential(
            SpatialAttention(3)
            # nn.Conv2d(in_channels//2, in_channels, 1)  # 输出通道匹配输入
        )

        self.laplacian_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.SiLU(inplace=True),
            ) for _ in range(num_scales)
        ])

        self.fuse_c = ChannelAttention(in_channels)

    def forward(self, x):
        # 生成通道敏感的平滑掩码
        smooth_mask = self.smooth_conv(x)
        # print(smooth_mask.shape)
        # 多尺度拉普拉斯响应
        responses = []
        for i, laplacian in enumerate(self.laplacian_scales):
            scaled_mask = F.max_pool2d(smooth_mask, 2 ** i)
            response = laplacian(scaled_mask).abs()
            # print(response.shape)
            response = F.interpolate(response, x.shape[-2:], mode='bilinear')
            # print(response.shape)
            responses.append(response)

        # 通道自适应的动态融合
        fused_weight = self.fuse_c(x) # [S,C,160,160]
        # print(fused_weight.shape)
        laplacian_fused = sum(w * r for w, r in zip(fused_weight.unbind(0), responses))


        # 最终权重生成（通道维度保持）
        return smooth_mask * (1 - laplacian_fused)


class EMSSHead(nn.Module):
    def __init__(self, in_channels,numscsle):
        super().__init__()
        self.emss = EMSS(in_channels,numscsle)
        # 跨通道融合（替代原残差连接）
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        weight = self.emss(x)
        return  self.fusion(torch.cat([weight,x],1))

if __name__ == '__main__':
    input = torch.randn(32,16,64,64)
    layer = EMSSHead(16,2)
    output = layer(input)
    print(output.size())
