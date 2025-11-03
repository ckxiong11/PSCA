import torch
import torch.nn as nn
import torchvision


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out




class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.proj_1(x)
        x = self.activation(x)
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn

# x = torch.randn(1, 256, 80, 80)  # B, C, H, W 其中 C 必须是 4 的倍数
# block = LKA(256)
# out = block(x)
# print(out.shape)

