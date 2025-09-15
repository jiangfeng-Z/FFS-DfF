"""
@author: zjf
@create time: 2024/4/26 15:17
@desc:
"""

import torch.nn as nn
import torch
import matplotlib
from Tool.check_and_create_folder import check_and_create_folder
from imageio import imwrite
from Model.Network import FFS_T
import numpy as np
import cv2 as cv
from Tool.adjust_range import adjust
from Tool.justpfm import write_pfm
import os
from matplotlib import cm
# 设置随机种子为10 确保每次输出保持一致

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def DepthtoFused_wyy(DepthImg, imgarr):
    h, w = DepthImg.shape
    DepthImg = np.round(DepthImg)
    count, rows, cols, channels = imgarr.shape
    Fused = np.zeros((h, w, channels), dtype=np.uint8)
    for k in range(w):
        for j in range(h):
            Dep = DepthImg[j, k]
            Fused[j, k, :] = imgarr[int(Dep - 1), j, k, :]
    return Fused


def main():
    model = FFS_T()
    model = model.to('cuda:0')
    pth_path_root = "../Train_M/Record/HCI_10_S/model_best.pth"
    save_root = "Record/deal_Hand/"
    check_and_create_folder(save_root)
    model = nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(pth_path_root)
    model.load_state_dict(checkpoint, strict=True)

    data_path_root = "/mnt/data2/zhangjiangfeng/zjf_data/PublicDataset/deal_Hand_Mobile_Datasets/"
    file_names = os.listdir(data_path_root)
    print(file_names)
    # amp
    model.eval()
    with torch.no_grad():
        for file_name in file_names:
            print(file_name)
            image_paths_root = data_path_root + file_name + "/"
            image_paths = os.listdir(image_paths_root)
            image_paths = sorted(image_paths)
            images = []
            for i in range(len(image_paths) - 1):
                image = cv.imread(image_paths_root + image_paths[i])
                images.append(image)
            images = np.array(images)
            images = images[::-1, :, :, :]
            images_aif = images
            images = images / 127.5 - 1.0
            N, H, W, C = images.shape
            images = np.transpose(images, (3, 0, 1, 2))
            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w = 0
            images = np.pad(images, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=(-1, -1))
            FS = torch.Tensor(np.expand_dims(images, axis=0))
            focus_dists = np.linspace(1, N, N)
            focus_dists = np.expand_dims(focus_dists, axis=1)
            focus_dists = np.expand_dims(focus_dists, axis=2)
            focus_dists = torch.Tensor(np.tile(focus_dists, [1, H + pad_h, W + pad_w]))
            focus_dists = focus_dists.unsqueeze(0)

            pred_l = model(FS, focus_dists)
            pred_l = np.squeeze(pred_l.data.cpu().numpy())
            pred_l = pred_l[:H, :W]

            AiF = DepthtoFused_wyy(pred_l, images_aif)
            cv.imwrite(save_root + str(file_name) + "_AiF.bmp", AiF)

            pred_l = adjust(pred_l, 0, 1)
            np.save(save_root + str(file_name) + "_pre.npy", pred_l)
            cv.imwrite(save_root + str(file_name) + "_pre.bmp", pred_l*255)
            write_pfm(save_root + str(file_name) + "_pre.pfm", pred_l)

            cmap = cm.get_cmap('jet')
            color_img = cmap(pred_l)[:, :, 0:3]
            # print(color_img.shape)
            color_img = (color_img * 255).astype(np.uint8)
            imwrite(save_root + str(file_name) + "_pre_c.jpg", color_img, quality=100)
            # cv.imwrite(root + str(idx) + "_pre_c_.png", cv.applyColorMap(adjust(np.squeeze(test_pred3), 0, 255).astype(np.uint8), cv.COLORMAP_JET))


if __name__ == "__main__":
        main()
