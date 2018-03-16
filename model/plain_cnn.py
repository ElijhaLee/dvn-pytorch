#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .layers import *


class PlainCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(PlainCNN, self).__init__()
        self.down1 = ConvDown(n_channels + n_classes, 32)
        self.down2 = ConvDown(32, 64)
        self.down3 = ConvDown(64, 128)
        self.down4 = ConvDown(128, 256)
        self.down5 = ConvDown(256, 512)
        self.down6 = ConvDown(512, 1024)

        self.fc1 = nn.Linear(4 * 4 * 1024, 1)
        # self.fc1 = nn.Linear(4 * 4 * 1024, 4 * 1024)
        # self.fc2 = nn.Linear(4 * 1024, 1)

    def forward(self, x):  # 256,5
        x = self.down1(x)  # 128,32
        x = self.down2(x)  # 64,64
        x = self.down3(x)  # 32,128
        x = self.down4(x)  # 16,256
        x = self.down5(x)  # 8,512
        x = self.down6(x)  # 4,1024

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)

        iou_pred = F.tanh(x) / 2 + 0.5

        return iou_pred
