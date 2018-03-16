import torch
from torch.nn import Module
from model.plain_cnn import PlainCNN

n_img_c = 3
n_class = 2


class ValueNet(Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = PlainCNN(n_channels=n_img_c, n_classes=n_class)

    def forward(self, img, lbl):
        # img: n,3,h,w
        # lbl: n,n_class,h,w
        assert img.size(1) == n_img_c, 'img channel is not equal to %d' % n_img_c
        assert lbl.size(1) == n_class, 'lbl channel is not equal to %d' % n_class
        x = torch.cat([img, lbl], dim=1)
        return self.net(x)
