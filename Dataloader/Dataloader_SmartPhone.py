"""
@author: zjf
@create time: 2024/4/24 9:07
@desc:
"""
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
import cv2, h5py, os, math, time
from Dataloader.augmentation import *
from tqdm import tqdm


class Smartphone(Dataset):

    def __init__(self, mode, num_imgs):
        self.mode = mode
        self.num_imgs = num_imgs
        self.input_size = (504, 378)
        self.center_crop = (336, 252)
        self.rand_crop = (224, 224)
        self.cropping = (self.center_crop[0] - self.rand_crop[0], self.center_crop[1] - self.rand_crop[1])
        self.indexes = np.rint(np.linspace(0, 48, num_imgs, endpoint=True)).astype(np.uint8)
        self.focus_dists = []
        # https://storage.googleapis.com/cvpr2020-af-data/LearnAF%20Dataset%20Readme.pdf
        focus_dists = [3910.92, 2289.27, 1508.71, 1185.83, 935.91, 801.09, 700.37, 605.39, 546.23, 486.87, 447.99,
                       407.40, 379.91, 350.41, 329.95, 307.54,
                       291.72, 274.13, 261.53, 247.35, 237.08, 225.41, 216.88, 207.10, 198.18, 191.60, 183.96, 178.29,
                       171.69, 165.57, 160.99, 155.61, 150.59, 146.81,
                       142.35, 138.98, 134.99, 131.23, 127.69, 124.99, 121.77, 118.73, 116.40, 113.63, 110.99, 108.47,
                       106.54, 104.23, 102.01]
        for index in self.indexes:
            self.focus_dists.append(focus_dists[index])
        self.focus_dists = np.expand_dims(self.focus_dists, axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=2).astype(np.float32)
        self.focus_dists = self.focus_dists * 0.001
        self.Fovs = (1 / 0.00444) - (
                    1 / np.array(self.focus_dists))  # https://www.devicespecifications.com/en/model/121b4c25
        self.Fovs = self.Fovs / np.min(self.Fovs)
        self.Fovs = np.expand_dims(self.Fovs, axis=0)
        if mode == "train":
            self.focus_dists = torch.Tensor(np.tile(self.focus_dists, [1, self.rand_crop[0], self.rand_crop[1]]))
        elif mode == "test":
            self.focus_dists = torch.Tensor(
                np.tile(self.focus_dists, [1, self.center_crop[0] + 16, self.center_crop[1] + 4]))
        self.focus_dists = 1 / self.focus_dists
        self.max_depth = 1 / 0.10201
        self.min_depth = 1 / 3.91092
        self.root = '/mnt/data2/zhangjiangfeng/zjf_data/PublicDataset/SmartPhone/'
        self.depths = []
        self.confids = []
        self.FS = []
        if mode == "train":
            for i in tqdm(range(1, 8), desc="trainset"):
                path = self.root + mode + str(i) + '/'
                scenes = os.listdir(path + 'scaled_images/')
                for scene in scenes:
                    self.depths.append(path + 'merged_depth/' + scene + '/' + 'result_merged_depth_center.png')
                    self.confids.append(path + 'merged_conf/' + scene + '/' + 'result_merged_conf_center.exr')
                    FS_imgs = []
                    for j in self.indexes:
                        FS_imgs.append(
                            path + 'scaled_images/' + scene + '/' + str(j) + '/result_scaled_image_center.jpg')
                    self.FS.append(FS_imgs)
        elif mode == "test":
            path = self.root + mode + '/'
            scenes = os.listdir(path + 'scaled_images/')
            for scene in scenes:
                self.depths.append(path + 'merged_depth/' + scene + '/' + 'result_merged_depth_center.png')
                self.confids.append(path + 'merged_conf/' + scene + '/' + 'result_merged_conf_center.exr')
                FS_imgs = []
                for j in self.indexes:
                    FS_imgs.append(path + 'scaled_images/' + scene + '/' + str(j) + '/result_scaled_image_center.jpg')
                self.FS.append(FS_imgs)

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        # Create sample dict
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        # base_focal_length = self.FS_focal_length[idx][48]
        FS = np.zeros((self.center_crop[0], self.center_crop[1], self.num_imgs, 3), dtype=np.float32)
        for i in range(0, self.num_imgs):
            img = cv2.imread(self.FS[idx][i]).astype(np.float32)[:, :, :]
            FS[:, :, i, :] = img[84:-84, 63:-63, :].astype(np.float32)
        img = cv2.imread(self.FS[idx][self.num_imgs - 1]).astype(np.float32)[:, :, :]
        FS[:, :, (self.num_imgs - 1), :] = img[84:-84, 63:-63, :].astype(np.float32)

        gt = cv2.imread(self.depths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)[84:-84, 63:-63]
        gt = gt / 255.0
        gt = (20) / (100 - (100 - 0.2) * gt)
        gt = 1 / gt
        conf = cv2.imread(self.confids[idx], cv2.IMREAD_UNCHANGED)[84:-84, 63:-63, -1]
        conf[conf > 1.0] = 1.0

        if self.mode == "train":
            y_crop, x_crop, contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
            FS, gt, conf = randcrop_3d_w_conf(FS, gt, conf, x_crop, y_crop, self.cropping[1], self.cropping[0])
            FS = image_augmentation(FS, contrast, brightness, gamma)
            FS, gt, conf = horizontal_flip_w_conf(FS, gt, conf, flip_x)
            FS, gt, conf = vertical_flip_w_conf(FS, gt, conf, flip_y)
            FS, gt, conf = rotate_w_conf(FS, gt, conf, angle)

        elif self.mode == "test":
            FS = FS / 127.5 - 1.0
        gt[gt < self.min_depth] = 0.0
        gt[gt > self.max_depth] = 0.0

        mask = torch.from_numpy(np.where(gt == 0.0, 0., 1.).astype(np.bool_))

        FS = torch.from_numpy(np.transpose(FS, (3, 2, 0, 1)))
        N, C, H, W = FS.shape
        if H % 32 != 0:
            pad_h = 32 - (H % 32)
        else:
            pad_h = 0
        if W % 32 != 0:
            pad_w = 32 - (W % 32)
        else:
            pad_w = 0
        FS = F.pad(torch.Tensor(FS), (0, pad_w, 0, pad_h))  # top 4 padding

        gt = torch.from_numpy(gt)
        return FS, gt, self.focus_dists, mask, conf, torch.from_numpy(self.Fovs)

    def get_seeds(self):
        return (
            random.randint(0, self.cropping[0] - 1), random.randint(0, self.cropping[1] - 1), random.uniform(0.4, 1.6),
            random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0), random.uniform(0, 1.0),
            random.randint(0, 3))
