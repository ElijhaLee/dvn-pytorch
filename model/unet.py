#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .layers import *

nn.DataParallel
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.inc = ConvIn(n_channels, 64)
        self.down1 = ConvDown(64, 128)
        self.down2 = ConvDown(128, 256)
        self.down3 = ConvDown(256, 512)
        # self.down4 = ConvDown(512, 512)
        self.up1 = ConvUp(512, 512)
        self.up2 = ConvUp(512, 256)
        self.up3 = ConvUp(256, 128)
        # self.up4 = ConvUp(64, 64)
        self.outc = ConvOut(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # 32,512
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)  # 64,256
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        x = self.outc(x)
        return x
