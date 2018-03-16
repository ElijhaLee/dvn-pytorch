# a = onehot_label[3]
# b = a.view(2,256,256)
# import torchvision.transforms as t
# c = b.float()
# d = c.long()
# e = d.cpu()[1].numpy()
# import skimage.io as io
# import numpy as np
# io.imsave('qwe.jpg',e.astype(np.float))


import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
time.sleep(0.5)

n1 = nn.Linear(1000, 1000)
n2 = nn.Linear(1000, 1000)
n1.cuda()
n2.cuda()

for step in range(100000):
    i = Variable(torch.randn(32, 1000).cuda())
    o1 = n1(i)
    o11 = n1(o1)
    o1.detach_()
    o2 = n2(o11)
    loss = o2.sum()
    loss.backward()  # 显存不是在这释放的，即便不backward，显存也不会一直增加
    print(loss, step, '/', 100000)
