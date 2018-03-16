import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from collections import defaultdict
from torch.autograd.variable import Variable
import pickle as pkl
import os.path as osp


def get_img_nm(root):
    f_list = os.listdir(os.path.join(root))
    img_nms = []
    for f in f_list:
        if f.split('.')[-1] == 'jpg':
            img_nms.append(f)

    return img_nms


class DatasetHorse(Dataset):
    def __init__(self, data_root):
        self.image_dir = osp.join(data_root, 'rgb')
        self.label_dir = osp.join(data_root, 'figure_ground')
        self.img_list = get_img_nm(self.image_dir)
        self.seg_list = get_img_nm(self.label_dir)

        assert self.img_seg_match_assertion(), 'Different length of img_nms and seg_nms'

        self.matched_pair_list = self.pair_0()

        self.len = len(self.img_list)

    def __getitem__(self, idx):
        ret = self.get_matched_pair(idx)
        return ret

    def __len__(self):
        return self.len

    def pair_0(self):
        '''
        Pair image and label. Each pair is matched.
        :return: an image_label pair
        '''
        ret = list()
        for img_nm in self.img_list:
            img_no = int(img_nm[5:8])
            img_path = osp.join(self.image_dir, img_nm)
            lbl_path = osp.join(self.label_dir, img_nm)
            ret.append((img_nm, img_no, img_path, lbl_path))
        return ret

    def get_matched_pair(self, idx):
        return self.matched_pair_list[idx]

    def img_seg_match_assertion(self):
        img_nms = np.array(self.img_list)
        seg_nms = np.array(self.seg_list)
        bool_ = img_nms.sort() == seg_nms.sort()
        return bool_


if __name__ == '__main__':
    data_dir = '/data/yifanl/dvn/weizmann_horse_db/'
    dataset = DatasetHorse(data_dir)
    dl = DataLoader(dataset, 32, True)
    for d in dl:
        print(d)

    print('done')
