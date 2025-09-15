"""
@author: zjf
@create time: 2024/9/29 16:29
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
from Dataloader.Dataloader_SmartPhone import Smartphone
import cv2 as cv
from Tool.adjust_range import adjust
from Tool.justpfm import write_pfm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    model = FFS_T()

    model = model.to('cuda:0')


    pth_path_root = "../Train_S/Record/SmartPhone/model_best.pth"
    save_root = "Record/SmartPhone/"


    model = nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(pth_path_root)
    model.load_state_dict(checkpoint, strict=True)

    num_test = 47
    Height = 336
    Width = 252
    
    Dataset = Smartphone('test', 10)
    check_and_create_folder(save_root)
    dataloader = DataLoader(Dataset, 1, shuffle=False, num_workers=4, pin_memory=True)
    print(len(dataloader))
    # amp

    model.eval()
    with torch.no_grad():
        Avg_abs_rel = 0.0
        Avg_sq_rel = 0.0
        Avg_mse = 0.0
        Avg_mae = 0.0
        Avg_rmse = 0.0
        Avg_rmse_log = 0.0
        Avg_accuracy_1 = 0.0
        Avg_accuracy_2 = 0.0
        Avg_accuracy_3 = 0.0
        Avg_bumpiness = 0.0
        Avg_sc_inv = 0.0
        AVG_time = 0.0

        for idx, samples in enumerate(tqdm(dataloader, desc="Test")):
            test_input, test_gt_depth, test_focus_dists, test_mask, test_conf, _ = samples
            test_focus_dists = test_focus_dists.cuda()
            test_input = test_input.cuda()
            test_gt_depth = np.squeeze(test_gt_depth.numpy())
            test_conf = np.squeeze(test_conf.data.cpu().numpy())
            test_mask = np.squeeze(test_mask.data.cpu().numpy())
            max_depth = np.max(test_gt_depth[test_conf == 1.0])
            min_depth = np.min(test_gt_depth[test_conf == 1.0])

            start = time.time()
            pred = model(test_input, test_focus_dists)

            AVG_time = AVG_time + (time.time() - start)
            pred = np.squeeze(pred.data.cpu().numpy())
            pred = pred[:Height, :Width]

            Avg_abs_rel = Avg_abs_rel + mask_abs_rel(pred, test_gt_depth, test_mask)
            Avg_sq_rel = Avg_sq_rel + mask_sq_rel(pred, test_gt_depth, test_mask)
            Avg_mse = Avg_mse + mask_mse_w_conf(pred, test_gt_depth, test_conf, test_mask)
            Avg_mae = Avg_mae + mask_mae_w_conf(pred, test_gt_depth, test_conf, test_mask)
            Avg_rmse = Avg_rmse + mask_rmse(pred, test_gt_depth, test_mask)
            Avg_rmse_log = Avg_rmse_log + mask_rmse_log(pred, test_gt_depth, test_mask)
            Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(pred, test_gt_depth, 1, test_mask)
            Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(pred, test_gt_depth, 2, test_mask)
            Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(pred, test_gt_depth, 3, test_mask)
            Avg_bumpiness = Avg_bumpiness + get_bumpiness(test_gt_depth, pred, test_mask)
            pred_ = (pred - min_depth) / (max_depth - min_depth)
            test_gt_depth_ = (test_gt_depth - min_depth) / (max_depth - min_depth)
            Avg_sc_inv = Avg_sc_inv + sc_inv(pred_, test_gt_depth_, test_mask)
            np.save(save_root + str(idx) + "_pre.npy", pred)
            np.save(save_root + str(idx) + "_gt.npy", test_gt_depth)
            cv.imwrite(save_root + str(idx) + "_pre.bmp", adjust(pred, 0, 255))
            cv.imwrite(save_root + str(idx) + "_gt.bmp", adjust(test_gt_depth, 0, 255))
            write_pfm(save_root + str(idx) + "_pre.pfm", pred)
            write_pfm(save_root + str(idx) + "_gt.pfm", test_gt_depth)
            cmap = cm.get_cmap('jet')
            color_img = cmap((pred - min_depth) / (max_depth - min_depth))[:, :, 0:3]
            # print(color_img.shape)
            color_img = (color_img * 255).astype(np.uint8)
            imwrite(save_root + str(idx) + "_pre_c.jpg", color_img, quality=100)
            color_img_gt = cmap((test_gt_depth - min_depth) / (max_depth - min_depth))[:, :, 0:3]
            color_img_gt = (color_img_gt * 255).astype(np.uint8)
            imwrite(save_root + str(idx) + "_gt_c.jpg", color_img_gt, quality=100)
            # cv.imwrite(root + str(idx) + "_pre_c_.png", cv.applyColorMap(adjust(np.squeeze(test_pred3), 0, 255).astype(np.uint8), cv.COLORMAP_JET))

        Avg_abs_rel = Avg_abs_rel / num_test
        Avg_sq_rel = Avg_sq_rel / num_test
        Avg_mse = Avg_mse / num_test
        Avg_mae = Avg_mae / num_test
        Avg_rmse = Avg_rmse / num_test
        Avg_rmse_log = Avg_rmse_log / num_test
        Avg_accuracy_1 = Avg_accuracy_1 / num_test
        Avg_accuracy_2 = Avg_accuracy_2 / num_test
        Avg_accuracy_3 = Avg_accuracy_3 / num_test
        Avg_bumpiness = Avg_bumpiness / num_test
        Avg_sc_inv = Avg_sc_inv / num_test
        AVG_time = AVG_time / num_test
        with open(save_root + "metrics.txt", "w") as file:
            file.write("Avg_abs_rel : " + str(Avg_abs_rel) + "\n")
            file.write("Avg_sq_rel : " + str(Avg_sq_rel) + "\n")
            file.write("Avg_mse : " + str(Avg_mse) + "\n")
            file.write("Avg_mae : " + str(Avg_mae) + "\n")
            file.write("Avg_rmse : " + str(Avg_rmse) + "\n")
            file.write("Avg_rmse_log : " + str(Avg_rmse_log) + "\n")
            file.write("Avg_accuracy_1 : " + str(Avg_accuracy_1) + "\n")
            file.write("Avg_accuracy_2 : " + str(Avg_accuracy_2) + "\n")
            file.write("Avg_accuracy_3 : " + str(Avg_accuracy_3) + "\n")
            file.write("Avg_bumpiness : " + str(Avg_bumpiness) + "\n")
            file.write("Avg_sc_inv : " + str(Avg_sc_inv) + "\n")
            file.write("AVG_time : " + str(AVG_time) + "\n")
        metrics = {"Avg_abs_rel": Avg_abs_rel, "Avg_sq_rel": Avg_sq_rel, "Avg_mse": Avg_mse,
                   "Avg_mae": Avg_mae, "Avg_rmse": Avg_rmse, "Avg_rmse_log": Avg_rmse_log,
                   "Avg_accuracy_1": Avg_accuracy_1, "Avg_accuracy_2": Avg_accuracy_2, "Avg_accuracy_3": Avg_accuracy_3,
                   "Avg_bumpiness": Avg_bumpiness, "Avg_sc_inv": Avg_sc_inv, "AVG_time": AVG_time}
        io.savemat(save_root + "metrics.mat", metrics)

        print("Avg_mse : ", Avg_mse)


if __name__ == "__main__":
    main()
