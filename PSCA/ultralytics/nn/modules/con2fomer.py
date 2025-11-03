import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from .conv import Conv,DWConv
from timm.layers import DropPath, trunc_normal_
from .Attention import scalattention
from .wtconv import WTConv2d

class MLPs(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x

class MLPc(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        hidden_features = in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class stem_s(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        return x

class stem_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avg_pool1(x).flatten(2)
        return x

class ConvMods(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(1, eps=1e-6, data_format="channels_first")
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # self.stripConv1 =
        self.stems = stem_s()
        self.attenc = Attentionc(1,1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.GELU(),
            DWConv(dim, dim, 3, 1, 1)

        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.GELU(),
            DWConv(dim, dim, 5, 1, 2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.GELU(),
            DWConv(dim, dim, 7, 1, 3)
        )

        self.WTconv5 = nn.Sequential(
              nn.Conv2d(1, 1, 1, 1),
              nn.GELU(),
              WTConv2d(1, 1, 5),
                                   )

        self.v4 = nn.Conv2d(1, 1, 1)
        self.stems = stem_s()
        self.fusion = Conv(dim*3, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B,C,H,W = x.shape
        xs_ = self.stems(x)
        xs = self.norm1(xs_)
        xs = self.WTconv5(xs)
        awt = self.proj1(self.v4(xs_) * xs)
        wt = (awt * x) + x
        # wt = self.norm(wt)
        xc = self.stemc(wt)
        xc = self.norm1(xc)
        xc = self.attenc(xc)
        xc = xc.unsqueeze(-1)

        # x1, x2, x3, x4 = wt.split((C/4, C/4, C/4, C/4), 1)
        x1_1 = self.conv3(wt)
        x2_2 = self.conv5(wt)
        x3_3 = self.conv7(wt)
        fus = self.fusion(torch.cat((x1_1, x2_2, x3_3), dim=1))
        attention = self.proj(self.v(wt) * fus)+wt
        attention = self.norm(attention)
        return attention



class Attentionc(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _x = (attn @ v).transpose(1, 2).view(B, N, C)
        x = self.proj(_x)
        x = self.norm1(x + x1)
        # x = self.norm2(x + self.mlp(x))
        return x

class Block(nn.Module):
    def __init__(self, inchannel, drop_path=0.):
        super().__init__()

        self.attnwts = scalattention(inchannel)
        # self.attnc = Attentionc(1, 1, drop_path=drop_path)
        # self.norm1 = LayerNorm(1, data_format='channels_first')
        # self.norm2 = nn.LayerNorm(1)
        # self.stems = stem_s()
        # self.v = nn.Conv2d(inchannel, inchannel, 1, 1)
        # self.stemc = stem_c()
        # # self.raw_param1 = nn.Parameter(torch.randn(1))
        # # self.sigmoid = nn.Sigmoid()
        # # self.raw_param2 = nn.Parameter(torch.randn(1))
        # self.conv1 = Conv(inchannel, inchannel // 2, 1, 1)
        # self.conv2 = Conv(inchannel//2, inchannel, 1, 1)
        # self.proj1 = nn.Conv2d(inchannel//2,inchannel//2,1,1)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((inchannel)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((inchannel)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # xs = self.stems(x)
        attens = self.attnwts(x)
        # attens = (attens * x) + x
        # xc = self.stemc(attens)
        # xc = self.attnc(self.norm2(xc))
        # xc1 = xc.unsqueeze(-1)
        # attenc = (attens * xc1) + attens
        return attens


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

if __name__ == '__main__':
    a = torch.randn(1, 1024, 80, 80)
    net = Block(1024, 1)
    # conm = ConvMod(128)
    print(net(a).shape)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(net(a).shape)  # 应输出 torch.Size([3, 256, 80, 80])
    # print(net(a).shape)