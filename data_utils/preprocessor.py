from PIL import Image
import os
import sys
import data_utils.transforms as T
import torch

# import torchvision.transforms as T

sys.path.append(os.path.abspath('..'))


class Preprocessor:
    def __init__(self, dataset, width, height):
        # super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.width = width
        self.height = height
        self.transform = self.get_transoformer()

    def get_transoformer(self):
        transformer = T.Compose([
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),  # changed, add color jitter
            T.RandomSizedRectCrop(self.height, self.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        return transformer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        # if isinstance(indices, (tuple, list)):
        #     return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_nm, img_no, img_path, lbl_path = self.dataset[index]
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        torch.unsqueeze(lbl, dim=0, out=lbl)
        return img_nm, img_no, img, lbl


if __name__ == '__main__':
    from data_utils.dataset import DatasetHorse

    data_dir = '/data/yifanl/dvn/weizmann_horse_db/'
    dataset = DatasetHorse(data_dir)
    pre = Preprocessor(dataset, 100, 80)
