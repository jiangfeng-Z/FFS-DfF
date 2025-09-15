"""
@author: zjf
@create time: 2024/4/24 20:37
@desc:
"""

import torch
import random
from torch.utils.data import Dataset
import cv2, h5py, os
from Dataloader.augmentation import *
from tqdm import tqdm


class FlyingThings3D(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.train_size = (256, 256)
        self.num_imgs = 15
        self.input_size = (540, 960)
        self.cropping = (self.input_size[0] - self.train_size[0], self.input_size[1] - self.train_size[1])
        self.rgb_paths = [[] for i in range(self.num_imgs)]
        self.disp_paths = []
        self.low_bound = 10
        self.high_bound = 100
        self.focus_dists = np.linspace(self.low_bound, self.high_bound, self.num_imgs)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists, axis=2).astype(np.float32)
        with open("/mnt/data1/zhangjiangfeng/zjf_data/PublicDataset/FlyingThings3D_FS/" + mode + "/focal_stack_path.txt",
                  'r') as f:
            for line in tqdm(f.readlines(), desc="flyingthings"):
                tmp = line.strip().split()
                for i in range(self.num_imgs):
                    self.rgb_paths[i].append(tmp[i])
                self.disp_paths.append(tmp[-1])

    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self, idx):  # TEST/Train
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        depth = cv2.imread(self.disp_paths[idx], cv2.IMREAD_UNCHANGED)  # depth check range, shape

        imgs = np.concatenate([np.expand_dims(cv2.imread(x[idx]), axis=3) for x in self.rgb_paths], 3)  # C*H*W
        if self.mode == "train":
            y_crop, x_crop, contrast, brightness, gamma, flip_x, flip_y, angle = self.get_seeds()
            imgs, depth = randcrop_3d(imgs, depth, x_crop, y_crop, self.cropping[1], self.cropping[0])
            imgs = image_augmentation(imgs, contrast, brightness, gamma)
            imgs, depth = horizontal_flip(imgs, depth, flip_x)
            imgs, depth = vertical_flip(imgs, depth, flip_y)
            imgs, depth = rotate(imgs, depth, angle)
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists, [1, self.train_size[0], self.train_size[1]]))

            imgs = torch.Tensor(np.transpose(imgs, (2, 3, 0, 1)))
        elif self.mode == "val":
            imgs = imgs / 127.5 - 1.0
            imgs = torch.Tensor(np.transpose(imgs, (2, 3, 0, 1)))
            C, N, H, W = imgs.shape

            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w = 0

            imgs = np.pad(imgs, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=(-1, -1))
            Focus_Dists = torch.Tensor(
                np.tile(self.focus_dists, [1, self.input_size[0] + pad_h, self.input_size[1] + pad_w]))

            # pad_h = 32 -(self.input_size[0]%32) if self.input_size[0] >0 else 0

        depth[depth < 0.0] = 0.0
        mask = torch.from_numpy(np.where(depth == 0.0, 0., 1.).astype(np.bool_))
        depth = torch.Tensor(depth)

        return imgs, depth, Focus_Dists, mask

    def get_seeds(self):
        return (
            random.randint(0, self.cropping[0] - 1), random.randint(0, self.cropping[1] - 1), random.uniform(0.4, 1.6),
            random.uniform(-0.1, 0.1), random.uniform(0.5, 2.0), random.uniform(0, 1.0), random.uniform(0, 1.0),
            random.randint(0, 3))
