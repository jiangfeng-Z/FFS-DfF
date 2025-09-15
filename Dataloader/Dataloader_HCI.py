"""
@author: zjf
@create time: 2024/3/23 18:08
@desc:
"""

import torch
import random
from torch.utils.data import Dataset
import h5py
from Ablation.Dataloader_A.augmentation import *


class HCI_dataset(Dataset):

    def __init__(self, hdf5_filename, sample_count, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.input_size = (512, 512)
        self.sample_count = sample_count
        if stack_key == "stack_train":
            self.size = (256, 256)

        elif stack_key == "stack_val":
            self.size = (512, 512)

        self.cropping = (self.input_size[0] - self.size[0], self.input_size[1] - self.size[1])

        focus_dists = self.hdf5['focus_position_disp']
        focus_dists = np.squeeze(focus_dists, axis=0)
        focus_dists = np.expand_dims(focus_dists, axis=1)
        focus_dists = np.expand_dims(focus_dists, axis=2)

        self.focus_dists = torch.Tensor(np.tile(focus_dists, [1, self.size[0], self.size[1]]))
        self.min_dist = np.min(focus_dists)
        self.max_dist = np.max(focus_dists)

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        FS = self.hdf5[self.stack_key][idx].astype(np.float32)
        FS_re = np.zeros((512, 512, 3, self.sample_count), dtype=np.float32)
        for i in range(0, self.sample_count):
            FS_re[:, :, :, i] = FS[i, :, :, :]
        gt = self.hdf5[self.disp_key][idx].astype(np.float32)
        if self.stack_key == "stack_train":
            y_crop, x_crop, contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
            FS, gt = randcrop_3d(FS_re, gt, x_crop, y_crop, self.cropping[1], self.cropping[0])
            FS = image_augmentation(FS, contrast, brightness, gamma)
            FS, gt = horizontal_flip(FS, gt, flip_x)
            FS, gt = vertical_flip(FS, gt, flip_y)
            FS, gt = rotate(FS, gt, angle)
        elif self.stack_key == "stack_val":
            FS = FS_re / 127.5 - 1.0
            gt[gt < self.min_dist] = -3.0
            gt[gt > self.max_dist] = -3.0

        mask = torch.from_numpy(np.where(gt == -3.0, 0., 1.).astype(np.bool_))
        FS = torch.from_numpy(FS.transpose((2, 3, 0, 1)))
        gt = torch.from_numpy(gt)
        return FS, gt, self.focus_dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]

    def get_seeds(self):
        return (
        random.randint(0, self.cropping[0] - 1), random.randint(0, self.cropping[1] - 1), random.uniform(0.4, 1.6),
        random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0), random.uniform(0, 1.0),
        random.randint(0, 3))
