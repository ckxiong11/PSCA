import math

import numpy as np
import torch
import torch.nn as nn
from .conv import Conv,DWConv

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class nbt(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, num, shortcut=True, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.num = num
        # self.cv1x1 = nn.ModuleList(
        #     Conv(c1, c2 // num, 1) for i in range(num)
        # )
        self.cv1x1 = Conv(c1, c1 // num, 1)
        self.cv1x1 = Conv(c1, c1 // num, 1)
        self.cv3x3 = DWConv(c2 // num, c2//num, 3)
        self.cv3x3 = DWConv(c2 // num, c2//num, 3)

        self.cv3x3_1 = Conv(c2//num, c2 // num, 3)
        self.cv3x3_2 = Conv(c2//num, c2 // num, 3)

        # self.cv3x3 = nn.ModuleList(
        #     Conv(c1 // num, c2 // num, 3) for i in range(num)
        # )

        self.Con1x1 = Conv(c1, c2, 1)
        self.add = shortcut and c1 == c2

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x1 = self.cv1x1(x)
        x2 = self.cv1x1(x)
        B, C, H, W = x.shape
        # x1, x2 = x.split((C // 2, C // 2), dim=1)
        a = self.cv3x3(x1)
        b = self.cv3x3(x2)

        # c1 = self.cv3x3_1(a)
        # c2 = self.cv3x3_2(b)
        f1 = self.Con1x1(torch.cat((c1, c2), dim=1))

        return f1

if __name__ == '__main__':
    x = torch.rand(3, 64, 256, 256)
    model = nbt(64, 64, 2)
    out = model(x)
    print(out.shape)
