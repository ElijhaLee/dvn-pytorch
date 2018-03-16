import torch
import os
import os.path as osp
from data_utils.dataset import DatasetHorse
from data_utils.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from model.seg_net import SegNet
from model.value_net import ValueNet
from torch.autograd.variable import Variable
import os
import time
from trainer import Trainer
import tensorboardX as tb

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
time.sleep(0.2)

data_dir = '/data/yifanl/dvn/weizmann_horse_db/'
pic_width = 256
pic_height = 256
batch_size = 32
use_cuda = True
total_epoch = 1000
lr = 1e-3


def get_data():
    dataset = DatasetHorse(data_dir)
    preprocessor = Preprocessor(dataset, pic_width, pic_height)
    dl = DataLoader(preprocessor, batch_size, shuffle=True, num_workers=1)
    return dl


if __name__ == '__main__':
    dl = get_data()
    criterion_sn = torch.nn.NLLLoss2d()
    criterion_vn = torch.nn.MSELoss()

    # model
    sn = SegNet()
    vn = ValueNet()
    if use_cuda:
        sn.cuda()
        vn.cuda()

    opt_sn = torch.optim.Adam(sn.parameters(), lr=lr)
    opt_vn = torch.optim.Adam(vn.parameters(), lr=lr)

    trainer = Trainer(sn, vn, criterion_sn, criterion_vn)

    writer = tb.SummaryWriter('logs')

    # for d in dl:
    #     _, _, img, lbl = d
    #     img = Variable(img)
    #     lbl = Variable(lbl)
    #     if use_cuda:
    #         img = img.cuda()
    #         lbl = lbl.cuda()
    #
    #     img_res = model(img)
    for i in range(total_epoch):
        # if (i + 1) % 3 == 0:
        #     trainer.train(i, dl, None, opt_vn, writer=writer)
        # else:
        #     trainer.train(i, dl, opt_sn, None, writer=writer)
        trainer.train(i, dl, opt_sn, opt_vn, mode='vn', writer=writer)
