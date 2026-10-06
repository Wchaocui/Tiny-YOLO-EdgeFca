import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
torch.backends.cuda.enable_flash_sdp(enabled=True)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
def activation_function(act="RE"):
    res = nn.Hardswish()
    if act == "RE":
        res = nn.ReLU6(inplace=True)
    elif act == "GE":
        res = nn.GELU()
    elif act == "SI":
        res = nn.SiLU()
    elif act == "EL":
        res = nn.ELU()
    else:
        res = nn.Hardswish()
    return res

class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k == s else autopad(k, p, d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation_function(act) # nn.ReLU6(inplace=True) if act=="RE" else nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.c(x)))

class gamma_trans(nn.Module):
    def __init__(self, c1,c2):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, bias=True, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.ReLU(inplace=True)
        self.ave_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.pconv = nn.Conv2d(2*c1, c1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.max_pool(x)
        x1 = self.ave_pool(x1)
        x_m = torch.add(x,x2)
        x_a = torch.add(x1,x)
        x = torch.cat([x_m, x_a], dim=1)
        x = self.pconv(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x
class gamma_trans1(nn.Module):
    def __init__(self, c1,c2):
        super().__init__()

        # self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.Softsign()
        self.act1 = nn.SiLU(inplace=True)
        self.ave_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.pconv = nn.Conv2d(3*c1, c1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.ave_pool(x)
        x2 = self.max_pool(x)
        x2 = self.act(x2)

        x_m = torch.add(x,x2)
        x_a = torch.add(x1,x)
        x = torch.cat([x_m, x_a,x], dim=1)
        x = self.pconv(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        # self.k1 = nn.Sequential(nn.Conv1d(dim,dim,kernel_size=1, bias=qkv_bias),
        self.k1=                    nn.MaxPool2d(5,1,2)
        # self.k2 = nn.Sequential(nn.Conv1d(dim,dim,kernel_size=1,bias=qkv_bias),
        self.k2=                    nn.AvgPool2d(5,1,3)
        self.v = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)  # 替换为卷积层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1,bias=qkv_bias)  # 替换为卷积层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print(x.shape)
        B, N, C= x.shape
        # print(x.shape)
        # x: B,H*W,C_in-->B,N,C
        x = x.permute(0, 2, 1)  # 调整形状以适应卷积层-->B,C,N
        # print(x.shape)
        k1 = self.k1(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = self.k2(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #q,k,v-->B,N,h,d-->B,h,N,d
        # print(q.shape)
        attn1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q, k2.transpose(-2, -1)) * self.scale

        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        # attn-->B,h,N,N
        x1 = (attn1 @ v).transpose(1, 2).reshape(B, -1, C)
        # x-->B,h,N,d-->B,N,h,d-->B,N,C
        x2 = (attn2 @ v).transpose(1, 2).reshape(B, -1, C)
        # print(x.shape)
        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)
        x = torch.exp(x1-x2+0.00001)
        # x = torch.mul(x1,x2)
        x = self.proj(x)
        x= self.proj_drop(x)
        return x
class Attention1(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k1 = nn.Sequential(nn.MaxPool2d(5,1,2),
                                nn.Conv1d(dim,dim,kernel_size=1, bias=qkv_bias),)
        self.v = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)  # 替换为卷积层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1,bias=qkv_bias)  # 替换为卷积层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C= x.shape
        # x: B,H*W,C_in-->B,N,C
        x = x.permute(0, 2, 1)  # 调整形状以适应卷积层-->B,C,N
        k1 = self.k1(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #q,k,v-->B,N,h,d-->B,h,N,d
        attn1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.attn_drop(attn1)
        # attn-->B,h,N,N
        x1 = (attn1 @ v).transpose(1, 2).reshape(B, -1, C)
        # x-->B,h,N,d-->B,N,h,d-->B,N,C
        x1 = x1.permute(0,2,1)
        x = self.proj(x1)
        x= self.proj_drop(x)
        return x
class cMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,N,C = x.shape
        # print(x.shape)
        x = x.reshape(B,C,N)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = cMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(x.shape)
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        #-->(B, C, H*W)-->(B,N,C)
        x = self.drop_path(self.attn(self.norm1(x)))
        res =x
        # x-->(B,C,N)
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.mlp(self.norm2(x)))#B,C,N
        # print(x.shape)
        x =res + x
        return x #B,C,N


class gamma_trans3(nn.Module):
    def __init__(self, c1,c2,s=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, kernel_size=3, stride=2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU(),
            nn.Conv2d(c2 // 2, c2 , kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.global_net = SABlock(c2,num_heads=4)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):

        x = self.conv(x)
        # print(x.shape)
        h, w = x.size(2), x.size(3)
        x = self.global_net(x)#BCN
        # b,c,n = x.shape
        # print(b,c,n)
        x = x.reshape(x.shape[0], -1,h,w)
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape)
        return x



class stem_conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, act="SI"):
        super().__init__()
        self.conv = nn.Sequential(
            mn_conv(c1, 32*3,k=k,s=s,act=act),
            mn_conv(32*3, c2,1,1,act),
        )
        self.gamma = gamma_trans3(c1,c2,s=2)
    def forward(self, x):
        x1 = self.conv(x)
        # print(x1.shape)#b,c,h//2,w//2
        x2 = self.gamma(x)

        x = torch.add(x1,x2)
        return x


#
if __name__ == '__main__':
    x = torch.randn(64,96,40,40)
    # gamma = gamma_trans3(3,16)
    # out = gamma(x)
    stem = stem_conv(96,32,3,2,act="SI")
    out = stem(x)
    print(out.shape)


