import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_
from .conv import Conv

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B*nW, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return x


def window_reverse(windows, window_size, H, W, C):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x

class stem_s(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1.mean(dim=1, keepdim=True)
        return x1

class stem_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.avg_pool1(x).flatten(2)
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


    def forward(self, x1):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _x = (attn @ v).transpose(1, 2).view(B, N, C)
        x = self.proj(_x)
        x = self.norm1(x1 + x).unsqueeze(-1)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=1,shift_size=0,):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 动态计算 shift_size
        shift_size = self.shift_size if H > self.window_size and W > self.window_size else 0

        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = self.norm1(x).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))  # Pad right and bottom
        _, _, Hp, Wp = x.shape

        x_windows = window_partition(x, self.window_size)  # [B*num_windows, ws*ws, C]
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp, C)

        x = x[:, :, :H, :W]  # Remove padding

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))

        #MLP
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm2(x)
        x = x.view(B, H * W, C)
        x = self.mlp(x).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x += shortcut
        return x


class SVITBlock(nn.Module):
    def __init__(self, inchannel, dim, window_size, num_heads, drop_path=0.):
        super().__init__()
        self.stemc = stem_c()
        self.stems = stem_s()
        self.catten = Attention(dim, num_heads, drop_path)
        self.Conv = Conv(inchannel, inchannel, 1, 1)
        self.satten1 = SwinBlock(
            dim,
            window_size,
            num_heads,
            0
        )
        self.satten2 = SwinBlock(
            dim,
            window_size,
            num_heads,
            3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identy = x
        xs = self.stems(x)
        xc = self.stemc(x)
        attens = self.satten1(xs)
        attens = self.satten2(attens)
        attenc = self.catten(xc)
        # fu = self.sigmoid(self.Conv(attens*attenc))
        attention = identy + (x * attens * attenc)
        return attention


if __name__ == '__main__':
    x = torch.rand(3, 256, 80, 80)
    model = SVITBlock(256, 1, 7, 1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(model(x).shape)  # 应输出 torch.Size([3, 256, 80, 80])
