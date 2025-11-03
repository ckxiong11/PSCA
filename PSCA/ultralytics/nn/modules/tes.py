import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors


from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect"


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes

        self.nl = len(ch)  # number of detection layers
        ch = [x // 2 for x in ch]
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )


    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        split_feats = [(torch.chunk(f, 2, dim=1)) for f in x]
        x1 = [split[0] for split in split_feats]
        x2 = [split[1] for split in split_feats]

        x3 = []
        for i in range(self.nl):
            x1[i] = torch.cat((self.cv2[i](x1[i]), self.cv3[i](x1[i])), 1)
            x2[i] = self.cv2[i](x2[i])
            x3.append(torch.cat((x1[i], x2[i]), 1))
        if self.training:  # Training path
            return x3

if __name__ == '__main__':
    feats = [
        torch.randn(1, 256, 20, 20),  # 特征图 1
        torch.randn(1, 512, 40, 40),  # 特征图 2
        torch.randn(1, 1024, 80, 80),  # 特征图 3
    ]

    model = Detect(80,[256,512,1024])
    print(model(feats))