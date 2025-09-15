"""
@author: zjf
@create time: 2024/4/26 15:17
@desc:
"""

import torch.nn as nn
from scipy import io
import time
from matplotlib import cm
from Tool.check_and_create_folder import check_and_create_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from Tool.metrics import *
from imageio import imwrite
from Model.Network import FFS_T
from Dataloader.Dataloader_HCI import HCI_dataset
import cv2 as cv
from Tool.adjust_range import adjust
from Tool.justpfm import write_pfm
import os

# 设置随机种子为10 确保每次输出保持一致

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

min_depth = 1.0
max_depth = 100.0

def sample_index(sample, interval):
    # 多景深图像序列数
    temp_count = []
    for i in range(0, sample, interval):
        temp_count.append(i)
    step = len(temp_count)
    index = torch.linspace(0, sample - 1, steps=step)
    index = torch.round(index).to(dtype=torch.long)
    return interval, index

def count_bmp_files(directory):
    return len([f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.bmp')])

def extract_focused_sequence(image_sequence, patch_size=64):
    """
    提取多景深序列中，具有聚焦区域的中间部分。

    参数:
        image_sequence: List[np.ndarray]，RGB图像序列 (H, W, 3)
        patch_size: 区域大小，默认64

    返回:
        focused_sequence: List[np.ndarray]，剔除前后未聚焦帧后的中间图像序列
    """
    num_images = len(image_sequence)
    h, w, _ = image_sequence[0].shape
    h_patches = h // patch_size
    w_patches = w // patch_size

    # 存储每个图像中每个patch的清晰度值（使用Sobel边缘强度）
    focus_measure = np.zeros((num_images, h_patches, w_patches), dtype=np.float32)

    for idx, img in enumerate(image_sequence):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for i in range(h_patches):
            for j in range(w_patches):
                patch = gray[i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size]
                sobel_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
                sobel_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                focus_measure[idx, i, j] = magnitude.mean()

    # 找到每个区域的最大清晰度帧（即最聚焦）
    max_focus_indices = np.argmax(focus_measure, axis=0)

    # 标记每一帧是否在任何一个patch上是最聚焦帧
    is_focused_frame = np.zeros(num_images, dtype=bool)
    for i in range(h_patches):
        for j in range(w_patches):
            is_focused_frame[max_focus_indices[i, j]] = True

    # 找到聚焦区域帧的起止索引
    focus_indices = np.where(is_focused_frame)[0]
    if len(focus_indices) == 0:
        return []  # 没有任何聚焦图像

    start_idx = focus_indices[0]
    end_idx = focus_indices[-1] + 1  # 包含end_idx

    return image_sequence[start_idx:end_idx]


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


def main(interval):
    model = FFS_T()

    model = model.to('cuda:0')
    model_theta = "Hander_S"
    pth_path_root = "../Train_M/Record/" + model_theta + "/model_best.pth"
    save_root = "Record/Intaglio_Printing_Plates/" + model_theta + "/" + str(interval) + "/"
    check_and_create_folder(save_root)
    model = nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(pth_path_root)
    model.load_state_dict(checkpoint, strict=True)
    data_path_root = "/mnt/data2/zhangjiangfeng/zjf_data/PublicDataset/Intaglio_Printing_Plates/"

    # amp

    model.eval()
    with torch.no_grad():
        for file_name in range(11, 37 + 1):
            image_path = data_path_root + str(file_name) + "/src/"
            num = count_bmp_files(image_path)
            images = []
            # *************************************
            start_time = time.time()
            # *************************************
            for i in range(1, num + 1):
                image = cv.imread(image_path + str(i) + ".bmp")
                images.append(image)

            images = extract_focused_sequence(images)
            images = np.array(images)
            images_aif = images
            images = images / 127.5 - 1.0
            N, H, W, C = images.shape
            print(file_name, N)
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
            # 大于这个数字才进行采样
            if N / interval >= 2:
                test_interval, test_index = sample_index(N, interval)
                FS = FS[:, :, ::test_interval, :, :]
                focus_dists = focus_dists[:, test_index, ...]
            pred_l = model(FS, focus_dists)
            pred_l = pred_l[:H, :W]
            pred_l = np.squeeze(pred_l.data.cpu().numpy())
            end_time = time.time()
            # *************************************
            total_time = end_time - start_time
            print(total_time)
            break
            # *************************************
            AiF = DepthtoFused_wyy(pred_l, images_aif)
            cv.imwrite(save_root + str(file_name) + "_AiF.bmp", AiF)

            pred_l = (pred_l - min_depth) / (max_depth - min_depth)
            np.save(save_root + str(file_name) + "_pre.npy", pred_l)
            cv.imwrite(save_root + str(file_name) + "_pre.bmp", pred_l * 255)
            write_pfm(save_root + str(file_name) + "_pre.pfm", pred_l)

            cmap = cm.get_cmap('jet')
            color_img = cmap(1 - pred_l)[:, :, 0:3]
            # print(color_img.shape)
            color_img = (color_img * 255).astype(np.uint8)
            imwrite(save_root + str(file_name) + "_pre_c.jpg", color_img, quality=100)
            # cv.imwrite(root + str(idx) + "_pre_c_.png", cv.applyColorMap(adjust(np.squeeze(test_pred3), 0, 255).astype(np.uint8), cv.COLORMAP_JET))


if __name__ == "__main__":
    for i in range(2, 16):
        main(interval=i)
