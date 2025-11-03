import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
from .wtconv import WTConv2d

class LayerNorm(nn.Module):
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


class MLPs(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        i_feats = 4 * n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 3, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1)

    def forward(self, x):

        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x * self.scale + shortcut

class scalattention(nn.Module):
    def __init__(self, in_channel):
        super(scalattention, self).__init__()
        midc = in_channel // 4
        self.norm1 = LayerNorm(in_channel, data_format='channels_first')
        self.norm = LayerNorm(midc, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(midc, midc, 1, 1),
            nn.GELU(),
            nn.Conv2d(midc, midc, 3, 1, 1, groups=midc)

        )
        self.v1 = nn.Conv2d(midc, midc, 1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(midc, midc, 1, 1),
            nn.GELU(),
            nn.Conv2d(midc, midc, 5, 1, 2, groups=midc)
        )
        self.v2 = nn.Conv2d(midc, midc, 1)

        self.conv7 = nn.Sequential(
            nn.Conv2d(midc, midc, 1, 1),
            nn.GELU(),
            nn.Conv2d(midc, midc, 7, 1, 3, groups=midc),
        )
        self.v3 = nn.Conv2d(midc, midc, 1)

        self.WTconv5 = nn.Sequential(
              nn.Conv2d(midc, midc, 1, 1),
              nn.GELU(),
              WTConv2d(midc, midc, 5),
                                   )

        self.v4 = nn.Conv2d(midc, midc, 1)

        self.proj1 = nn.Conv2d(in_channel, in_channel*2, 1, 1)
        self.fus = Conv(in_channel, in_channel, 1)
        self.v = nn.Conv2d(in_channel, in_channel, 1)
        self.proj = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm1(x)
        x = self.proj1(x)
        a, x = torch.chunk(x, 2, dim=1)
        # c = C // 4
        # x1, x2, x3, x4= self.covnsplit(x).split((1, 1, 1, 1), 1)
        x1, x2, x3, x4 = torch.chunk(a, 4, dim=1)
        x1 = self.norm(x1)
        x1_1 = self.conv3(x1)
        at1 = self.v1(x1) * x1_1

        x2 = self.norm(x2)
        x2_2 = self.conv5(x2)
        at2 = self.v2(x2) * x2_2

        x3 = self.norm(x3)
        x3_3 = self.conv7(x3)
        at3 = self.v3(x3) * x3_3

        x4 = self.norm(x4)
        x4_4 = self.WTconv5(x4)
        at4 = self.v4(x4) * x4_4

        fus = torch.cat((at1, at2, at3, at4), dim=1)
        aten = self.proj(self.v(x) * fus) * self.scale + shortcut
        return aten

# x = torch.randn((4, 1024, 80, 80))
# model = scalattention(1024)
# out = model(x)
# print(out.shape)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable parameters: {total_params}")
# print(model(x).shape)  # 应输出 torch.Size([3, 256, 80, 80])