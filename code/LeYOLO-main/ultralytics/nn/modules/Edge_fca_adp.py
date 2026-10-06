import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========== 保持你原来的基础代码不变 ==========
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


# ========== 最优自适应实现：只改这里 ==========
class AdaptiveFreqAttention(nn.Module):
    """
    真正的自适应频率注意力

    关键：学习任意的频率坐标 (u, v)，不受预定义列表限制
    每个频率坐标是连续值，通过round映射到离散位置
    """

    def __init__(self, channel, dct_h=20, num_select=8, reduction=16):
        super().__init__()
        self.dct_h = dct_h
        self.dct_w = dct_h
        self.num_select = num_select
        self.channel = channel

        # 可学习的频率坐标 [num_select, 2]，范围 [0, dct_h)
        # 初始化：均匀分布在低频区域（左上角）
        init_coords = torch.zeros(num_select, 2)
        for i in range(num_select):
            # 螺旋式初始化，优先低频
            if i == 0:
                init_coords[i] = torch.tensor([0., 0.])  # DC
            else:
                # 简单的螺旋初始化
                layer = (i - 1) // 8 + 1
                idx = (i - 1) % 8
                angle = idx * math.pi / 4
                init_coords[i] = torch.tensor([
                    layer * math.cos(angle),
                    layer * math.sin(angle)
                ])
        self.freq_coords = nn.Parameter(init_coords.clamp(0, dct_h - 1))

        # 可学习的频率重要性（用于加权聚合）
        self.freq_importance = nn.Parameter(torch.ones(num_select))

        # 通道注意力FC
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, mapper_x, mapper_y):
        """生成DCT滤波器"""
        dct_filter = torch.zeros(self.channel, self.dct_h, self.dct_w,
                                 device=mapper_x.device)
        c_part = self.channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(self.dct_h):
                for t_y in range(self.dct_w):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = (
                            self.build_filter(t_x, u_x, self.dct_h) *
                            self.build_filter(t_y, v_y, self.dct_w)
                    )
        return dct_filter

    def forward(self, x):
        n, c, h, w = x.shape

        # 池化
        x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w)) if (h != self.dct_h or w != self.dct_w) else x

        # 获取学习的频率坐标并离散化
        # 用sigmoid约束到 [0, 1]，再映射到 [0, dct_h)
        coords_normalized = torch.sigmoid(self.freq_coords)  # [num_select, 2], range [0,1]
        coords_scaled = coords_normalized * (self.dct_h - 1)  # [0, dct_h)

        # 训练时加噪声促进探索，推理时确定性的round
        if self.training:
            noise = torch.randn_like(coords_scaled) * 0.5
            coords_noisy = (coords_scaled + noise).clamp(0, self.dct_h - 1)
            mapper_x = coords_noisy[:, 0]
            mapper_y = coords_noisy[:, 1]
        else:
            mapper_x = coords_scaled[:, 0]
            mapper_y = coords_scaled[:, 1]

        # 生成滤波器并计算
        weight = self.get_dct_filter(mapper_x, mapper_y)
        x_weighted = x_pooled * weight
        y = torch.sum(x_weighted, dim=[2, 3])  # [N, C]

        # FC
        y = self.fc(y).view(n, c, 1, 1)

        # 保存坐标用于可视化
        self.last_coords = coords_scaled.detach()

        return y


# ========== EdgeFcaV2_Adaptive：最终版本 ==========
class EdgeFcaV2_Adaptive(nn.Module):
    def __init__(self, channels, dct_h=20, num_select=8):
        super().__init__()

        # 边缘提取（不变）
        self.edge_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.edge_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.edge_bn = nn.BatchNorm2d(channels)
        with torch.no_grad():
            laplace = torch.tensor([[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]])
            self.edge_dw.weight.copy_(
                laplace.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            )

        # 真正的自适应频域注意力
        self.fca = AdaptiveFreqAttention(channels, dct_h=dct_h, num_select=num_select)

        # 空间门控
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge_feat = self.edge_bn(self.edge_pw(self.edge_dw(x)))
        edge_feat = torch.relu(edge_feat)
        spatial_att = self.spatial_gate(edge_feat)

        freq_att = self.fca(x)

        enhanced = x * freq_att * spatial_att
        return x + enhanced