import torch
from torch.nn import Module
from model.unet import UNet


class SegNet(Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.net = UNet()

    def forward(self, x):
        return self.net(x)
