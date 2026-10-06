import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class CTopKPooling(nn.Module):
    def __init__(self, ratio):
        super(CTopKPooling, self).__init__()
        self.ratio = ratio#缩小尺度
        self.fc = nn.Linear(1, 1)  # 可学习的权重向量

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        self.k = int(self.ratio * channels)

        # 使用全局平均池化和激活函数来计算得分
        scores = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        # print("scores1.size",scores.size())[batch,channels]
        scores = self.fc(scores.unsqueeze(-1)).squeeze(-1)  # 应用可学习的权重
        # print("scores2.size",scores.size())[batch,channels]
        # 应用softmax获取每个通道的权重
        scores = F.softmax(scores, dim=0)
        # print("scores3.size", scores.size())[batch,channels]
        # 选择Top-K特征图
        top_k_features = x * scores.unsqueeze(-1).unsqueeze(-1)  # 将权重应用到每个通道

        # 将Top-K特征图拼接
        pooled_features, _ = torch.topk(top_k_features.view(batch_size, channels, -1), self.k, dim=1)

        # 将Top-K特征图拼接
        pooled_features = pooled_features.view(batch_size, self.k, height, width)

        return pooled_features

class AvgTopKPool2d(nn.Module):
    def __init__(self, k, kernel_size):
        super(AvgTopKPool2d, self).__init__()
        self.k = k
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 计算填充量
        padding_height = (self.kernel_size - height % self.kernel_size) % self.kernel_size
        padding_width = (self.kernel_size - width % self.kernel_size) % self.kernel_size

        # 添加填充
        x = F.pad(x, (0, padding_width, 0, padding_height))

        # 调整特征图尺寸以适应kernel_size
        new_height = height + padding_height
        new_width = width + padding_width
        x = x.view(batch_size, channels, new_height // self.kernel_size, self.kernel_size, new_width // self.kernel_size, self.kernel_size)

        # 重新排列维度以便于TopK操作
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels, -1, self.kernel_size * self.kernel_size)

        # 在每个区域内进行TopK操作
        _, indices = torch.topk(x, self.k, dim=-1)

        # 根据索引选择TopK值
        topk_values = torch.gather(x, -1, indices)

        # 计算每个区域的TopK值的平均
        pooled = topk_values.view(batch_size, channels, -1, self.k).mean(dim=-1)

        # 调整输出形状以匹配原始高度和宽度
        pooled = pooled.view(batch_size, channels, new_height // self.kernel_size, new_width // self.kernel_size)

        # 去除填充
        pooled = pooled[:, :, :height // self.kernel_size, :width // self.kernel_size]
        #output.size = [b,c,h/k,w/k]
        return pooled

from ultralytics.nn.modules.conv import autopad
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class TOPKSPP(nn.Module):
    def __init__(self, c1, c2, k=5,kernel_size=5):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 1, 1 )
        self.m = AvgTopKPool2d(k,kernel_size)

    def forward(self, x):
        x = self.cv1(x)
        x = self.m(x)
        x = self.cv2(x)
        return x

if __name__ == '__main__':
    # topk = TOPKSPP(32,64)
    x = torch.rand(1,512,20,20)
    # 假设conv_output是卷积层输出的特征图
    # conv_output = torch.randn(10, 64, 32, 32)  # 示例数据
    #
    # # 创建TopK池化层实例，假设我们选择Top-50%特征图
    # topk_pool = CTopKPooling(ratio=0.7)
    # # 应用TopK池化
    # pooled_output = topk_pool(conv_output)
    # print(pooled_output.size())  # 输出的形状将保持为[batch_size, k, height, width]
    # # summary(topk_pool, (conv_output.shape))
    #
    # k = 4  # TopK的k值
    # kernel_size = 4  # 划分区域的大小
    #
    # # 假设输入特征图的尺寸为 (batch_size, channels, height, width)
    # batch_size, channels, height, width = 5, 3, 12, 12  # 示例尺寸
    #
    # # 创建一个随机特征图
    # x = torch.randn(batch_size, channels, height, width)
    #
    # # 创建Avg-TopK池化层
    # avg_topk_pool = AvgTopKPool2d(k,kernel_size)
    #
    # # 应用池化
    # pooled_x = avg_topk_pool(x)
    # print(pooled_x.size())