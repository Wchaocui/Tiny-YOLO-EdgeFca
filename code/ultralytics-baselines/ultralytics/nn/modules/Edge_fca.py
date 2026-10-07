import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========== 复用你的DCT基础代码（需小改支持外部传入method） ==========
def get_freq_indices(method):
    if method == 'top8_low4':
        # top8: 8个高频 + low4: 4个低频 = 12个频率分量
        tx, ty = get_freq_indices('top8')
        lx, ly = get_freq_indices('low4')
        mapper_x = tx + lx
        mapper_y = ty + ly
        return mapper_x, mapper_y
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32',]
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


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4
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
                            self.build_filter(t_x, u_x, tile_size_x) *
                            self.build_filter(t_y, v_y, tile_size_y)
                    )
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_h=7, method='top4'):
        super().__init__()
        self.reduction = 16
        self.dct_h = dct_h
        self.dct_w = self.dct_h

        mapper_x, mapper_y = get_freq_indices(method)
        self.num_split = len(mapper_x)
        # 关键：将频率索引映射到当前dct_h尺寸（7x7空间对应原始索引）
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (self.dct_w // 8) for temp_y in mapper_y]
        # 修正：原始索引范围0~7，映射到 0 ~ dct_h-1
        # scale = (dct_h - 1) / 7.0
        # mapper_x = [temp_x * scale for temp_x in mapper_x]
        # mapper_y = [temp_y * scale for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, self.dct_w, mapper_x, mapper_y, channel)
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
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return y  # 只返回通道注意力权重，不乘x，交给外部融合


# ========== EdgeFcaV2：修复版核心模块 ==========
class EdgeFcaV2(nn.Module):
    def __init__(self, channels,dct_h=7):
        super().__init__()

        # 1. 可学习边缘提取（替代pywt小波，纯卷积，硬件友好）
        self.edge_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.edge_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.edge_bn = nn.BatchNorm2d(channels)
        # 初始化为拉普拉斯核（二阶微分，对弱边缘敏感，适合低光照）
        with torch.no_grad():
            laplace = torch.tensor([[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]])
            self.edge_dw.weight.copy_(
                laplace.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            )

        # 2. 精简频域注意力（只取topk频率，减少计算）
        self.fca = MultiSpectralAttentionLayer(channels, dct_h=dct_h, method=f'top8')

        # 3. 空间门控：边缘区域获得更强的频域通道增强
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape

        # 边缘分支：生成边缘响应图 [N, C, H, W]
        edge_feat = self.edge_bn(self.edge_pw(self.edge_dw(x)))
        edge_feat = torch.relu(edge_feat)
        # 空间注意力：边缘强的地方权重高 [N, 1, H, W]
        spatial_att = self.spatial_gate(edge_feat)

        # 频域分支：生成通道权重 [N, C, 1, 1]
        freq_att = self.fca(x)

        # 联合增强：边缘区域 × 频域通道增强；平坦区域保留原特征
        # 利用广播: [N,C,H,W] * [N,C,1,1] * [N,1,H,W]
        enhanced = x * freq_att * spatial_att

        # 残差融合：确保信息不丢失，增强部分作为增量
        # 这样即使edge_gate学到全0，模型也不会退化（退化为恒等映射）
        return x + enhanced

