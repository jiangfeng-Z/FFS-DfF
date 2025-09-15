"""
@author: zjf
@create time: 2024/4/22 20:03
@desc:
"""

import torch
import random
from torch.utils.data import Dataset
import cv2, h5py
from os import listdir
from os.path import isfile, join
from Dataloader.augmentation import *
import OpenEXR


class FS6_dataset(Dataset):
    def __init__(self, mode):
        self.root = "/mnt/data2/zhangjiangfeng/zjf_data/PublicDataset/DefocusNet_Dataset/" + mode + "/"
        self.imglist_all = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "Dpt.exr"]
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.max_depth = 3.0
        focus_dists = np.array([0.1, 0.15, 0.3, 0.7, 1.5])
        focus_dists = np.expand_dims(focus_dists, axis=1)
        focus_dists = np.expand_dims(focus_dists, axis=2).astype(np.float32)
        self.mode = mode
        self.Focus_Dists = torch.Tensor(np.tile(focus_dists, [1, 256, 256]))

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, index):
        img_dpt = self.read_dpt(self.root + self.imglist_dpt[index])

        contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
        img_index = index * 5
        mats_input = np.zeros((256, 256, 3, 0))
        for i in range(5):
            img = cv2.imread(self.root + self.imglist_all[img_index + i])
            mats_input = np.concatenate((mats_input, np.expand_dims(img, axis=-1)), axis=3)

        if self.mode == "train":
            mats_input = image_augmentation(mats_input, contrast, brightness, gamma)
            mats_input, img_dpt = horizontal_flip(mats_input, img_dpt, flip_x)
            mats_input, img_dpt = vertical_flip(mats_input, img_dpt, flip_y)
            mats_input, img_dpt = rotate(mats_input, img_dpt, angle)
            img_dpt[img_dpt < 0.0] = 0.0
            img_dpt[img_dpt > 2.0] = 0.0

        elif self.mode == "test":
            mats_input = mats_input / 127.5 - 1.0
            img_dpt[img_dpt < 0.1] = 0.0
            img_dpt[img_dpt > 1.5] = 0.0

        mats_input = np.transpose(mats_input, (2, 3, 0, 1))

        mask = torch.from_numpy(np.where(img_dpt == 0.0, 0., 1.).astype(np.bool_))

        img_dpt = torch.Tensor(img_dpt)
        mats_input = torch.Tensor(mats_input)

        return mats_input, img_dpt,  self.Focus_Dists, mask

    def read_dpt(self, img_dpt_path):
        dpt_img = OpenEXR.InputFile(img_dpt_path)
        dw = dpt_img.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        (r, g, b) = dpt_img.channels("RGB")
        dpt = np.fromstring(r, dtype=np.float16)
        dpt.shape = (size[1], size[0])
        return dpt

    def get_seeds(self):
        return (random.uniform(0.4, 1.6), random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0),
                random.uniform(0, 1.0), random.randint(0, 3))
