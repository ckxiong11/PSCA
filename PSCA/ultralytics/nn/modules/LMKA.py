import torch
import torch.nn as nn

from .deformer_LKA import LKA

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiKernelConvBlock, self).__init__()
        assert in_channels % 4 == 0, "输入通道数必须能被4整除"
        group_channels = in_channels // 4

        self.branch1_conv1x3 = DepthwiseSeparableConv(group_channels, group_channels, kernel_size=(1, 3),
                                                      padding=(0, 1))
        self.branch1_conv3x1 = DepthwiseSeparableConv(group_channels, group_channels, kernel_size=(3, 1),
                                                      padding=(1, 0))
        self.branch1_fuse = nn.Conv2d(group_channels * 2, group_channels, kernel_size=1, bias=False)

        # 其他三组：5x5, 7x7, 9x9使用轻量卷积
        self.branch2 = DepthwiseSeparableConv(group_channels, group_channels, kernel_size=5, padding=2)
        self.branch3 = DepthwiseSeparableConv(group_channels, group_channels, kernel_size=7, padding=3)
        self.branch4 = DepthwiseSeparableConv(group_channels, group_channels, kernel_size=9, padding=4)

        self.Lper = LKA(group_channels)
        # 可选：统一归一化 + 激活
        self.conv1x1 = nn.Conv2d(in_channels,in_channels,1, 1)
        self.v1 = nn.Conv2d(group_channels,group_channels,1,1)
        self.v2 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.v3 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.v4 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.p1 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.p2 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.p3 = nn.Conv2d(group_channels, group_channels, 1, 1)
        self.p4 = nn.Conv2d(group_channels, group_channels, 1, 1)

    def forward(self, x):
        # 按通道均分为4组
        groups = torch.chunk(x, 4, dim=1)

        # 第一组：方向卷积 + concat + 1x1 fuse
        g1_1 = self.branch1_conv1x3(groups[0])
        g1_2 = self.branch1_conv3x1(groups[0])
        g1 = torch.cat([g1_1, g1_2], dim=1)
        g1 = self.branch1_fuse(g1)

        # 其余三个分支直接卷积
        g2 = self.branch2(groups[1])
        g3 = self.branch3(groups[2])
        g4 = self.branch4(groups[3])

        # 通道拼接
        midout = g1 + g2 + g3 + g4
        atten = self.Lper(midout)
        ag1 = self.p1(self.v1(g1) * atten)
        ag2 = self.p2(self.v2(g2) * atten)
        ag3 = self.p3(self.v3(g3) * atten)
        ag4 = self.p4(self.v4(g4) * atten)
        out = self.conv1x1(torch.cat((ag1, ag2, ag3, ag4),dim=1))
        return out + x


# x = torch.randn(1, 256, 80, 80)  # B, C, H, W 其中 C 必须是 4 的倍数
# block = MultiKernelConvBlock(256)
# out = block(x)
# print(out.shape)