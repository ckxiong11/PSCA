import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

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

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PVT2FFN2(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act = nn.GELU()
        self.fc3 = nn.Linear(hidden_features, in_features)
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
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm1 = nn.LayerNorm(1)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
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


    def forward(self, x1, H, W):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _x = (attn @ v).transpose(1, 2).view(B, N, C)
        x = self.proj(_x)
        x = self.norm1(x1 + x)

        return x

class fusion(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
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


    def forward(self, x1, x2):
        # print(x1.shape)
        B, N, C = x1.shape
        B_p, N_p, C_p = x2.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x2).reshape(B_p, N_p, self.num_heads, C_p // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn.squeeze(1).permute(0, 2, 1)
        # atten1 = attn.sum(dim=3).permute(0, 2, 1)
        # atten2 = attn.sum(dim=2).permute(0, 2, 1)
        return attn


class stem_s(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1.mean(dim=1, keepdim=True)

        x1 = x1.flatten(2).transpose(1, 2)

        return x1

class stem_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.avg_pool1(x).flatten(2)
        return x

class AttenBlock(nn.Module):
    def __init__(self, inchannel, dim, num_heads, r=4, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.stems = stem_s()
        self.stemc = stem_c()
        self.norm1 = norm_layer(1)
        self.norm2 = norm_layer(1)
        self.norm3 = norm_layer(1)
        self.attn1 = Attention(dim, num_heads, drop_path=drop_path)
        self.attn2 = Attention(dim, num_heads, drop_path=drop_path)
        self.atten3 = fusion(dim, num_heads)
        self.Conv = Conv(inchannel, inchannel, 1, 1)
        self.Conv1 = nn.Conv2d(inchannel, inchannel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(inchannel)
        self.sigmoid = nn.Sigmoid()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
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
        identy = x
        B, C, H, W = x.shape
        xs = self.stems(x)
        xs = self.attn1(self.norm1(xs), H, W)
        xc = self.stemc(x)
        xc = self.attn2(self.norm1(xc), H, W)
        # atten = self.atten3(xs, xc)
        # atten = atten.view(B, C, H, W)
        # # x1 = self.Conv(x1)
        # out = identy + (identy * atten)
        xs = xs.permute(0, 2, 1).view(B, 1, H, W)
        xc = xc.unsqueeze(-1)
        #
        # fu = self.sigmoid(self.Conv(xs * xc))
        attention = identy + (xs * xc)
        return attention


if __name__ == '__main__':
    x = torch.rand(3, 64, 256, 256)
    model = AttenBlock(256, 1, 1, 4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(model(x).shape)


