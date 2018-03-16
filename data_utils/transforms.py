from __future__ import absolute_import

from PIL import Image
import random
import math
import torchvision.transforms.functional as F
import torch
import numpy as np


# class RectScale(object):
#     def __init__(self, height, width, interpolation=Image.BILINEAR):
#         self.height = height
#         self.width = width
#         self.interpolation = interpolation
#
#     def __call__(self, img):
#         w, h = img.size
#         if h == self.height and w == self.width:
#             return img
#         return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, img_interpolation=Image.BILINEAR, lbl_interpolation=Image.NEAREST):
        self.height = height
        self.width = width
        self.img_interpolation = img_interpolation
        self.lbl_interpolation = lbl_interpolation

    def get_params(self, img):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w, h = img.size[0], img.size[1]
        i = j = 0
        return i, j, w, h

    def __call__(self, img, lbl):
        lbl = quantize_lbl(lbl)
        lbl = resize_lbl_as_img(img, lbl)
        i, j, w, h = self.get_params(img)
        img = img.crop((j, i, i + w, j + h))
        lbl = lbl.crop((j, i, i + w, j + h))
        img = img.resize((self.width, self.height), self.img_interpolation)
        lbl = lbl.resize((self.width, self.height), self.lbl_interpolation)
        return img, lbl
        # original version
        # for attempt in range(10):
        #     area = img.size[0] * img.size[1]
        #     target_area = random.uniform(0.64, 1.0) * area
        #     aspect_ratio = random.uniform(2, 3)
        #
        #     h = int(round(math.sqrt(target_area * aspect_ratio)))
        #     w = int(round(math.sqrt(target_area / aspect_ratio)))
        #
        #     if w <= img.size[0] and h <= img.size[1]:
        #         x1 = random.randint(0, img.size[0] - w)
        #         y1 = random.randint(0, img.size[1] - h)
        #
        #         img = img.crop((x1, y1, x1 + w, y1 + h))
        #         assert (img.size == (w, h))
        #
        #         return img.resize((self.width, self.height), self.interpolation)
        #
        # # Fallback
        # scale = RectScale(self.height, self.width,
        #                   interpolation=self.interpolation)
        # return scale(img)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


class ToTensor(object):
    def __call__(self, img, lbl):
        return F.to_tensor(img), torch.from_numpy(np.array(lbl)).long()


def quantize_lbl(lbl):
    lbl_np = np.array(lbl)
    lbl_np = (lbl_np / 128).astype(np.uint8)
    lbl = Image.fromarray(lbl_np)
    return lbl


def resize_lbl_as_img(img, lbl, lbl_interpolation=Image.NEAREST):
    w, h = img.size[0], img.size[1]
    return lbl.resize((w, h), lbl_interpolation)
