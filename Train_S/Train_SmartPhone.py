"""
@author: zjf
@create time: 2024/2/25 17:07
@desc:
"""
import torch.nn as nn
from Tool.metrics import *
from Dataloader.Dataloader_SmartPhone import Smartphone
from Model.Network import FFS_T
from Tool.Loss_Function import Charbonnier_Conf_Loss, Edge_Conf_Loss, FFT_Loss
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

charbonnier_conf_loss = Charbonnier_Conf_Loss()
edge_conf_loss = Edge_Conf_Loss()
fft_loss = FFT_Loss()

# loss 比例关系
c = 1.0
e = 0.01
f = 0.01


def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--record_root', default="Record/SmartPhone/", type=str, help='save root')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=5000, type=int, help='max epoch')
    parser.add_argument('--load_epoch', default=1, type=int, help='load epoch')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')

    args = parser.parse_args()
    record_root = args.record_root
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    load_epoch = args.load_epoch
    # 参数设置
    writer = SummaryWriter(log_dir=record_root + "logs")

    max_Depth = 1 / 0.10201
    min_Depth = 1 / 3.91092
    num_train = 355
    num_val = 47
    Height = 336
    Width = 252

    # 总采样数

    Inf_loss = float('inf')
    model = FFS_T()
    train_dataset = Smartphone('train', 10)
    valid_dataset = Smartphone('test', 10)

    model = nn.DataParallel(model)
    if load_epoch:
        path = record_root + "model_best.pth"
        model.load_state_dict(torch.load(path))
    else:
        path = "Record/HCI/model_best.pth"
        model.load_state_dict(torch.load(path), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    model = model.cuda()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=4, pin_memory=True)
    # amp
    for epoch in range(load_epoch, load_epoch + max_epoch + 1):  # chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        if (epoch - load_epoch) >= 0:
            model.eval()
            with torch.no_grad():
                Avg_mse = 0.0
                Avg_abs_rel = 0.0
                Avg_accuracy_1 = 0.0
                for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
                    valid_input, valid_gt_depth, valid_focus_dists, valid_mask, valid_conf, _ = samples
                    valid_input = valid_input.cuda()
                    valid_focus_dists = valid_focus_dists.cuda()
                    valid_gt_depth = np.squeeze(valid_gt_depth.numpy())
                    valid_conf = np.squeeze(valid_conf.data.cpu().numpy())
                    valid_mask = np.squeeze(valid_mask.data.cpu().numpy())
                    pred = model(valid_input, valid_focus_dists)
                    pred = np.squeeze(pred.data.cpu().numpy())
                    pred = pred[:Height, :Width]
                    # print(pred.shape)
                    Avg_mse = Avg_mse + mask_mse_w_conf(pred, valid_gt_depth, valid_conf,valid_mask)
                    Avg_abs_rel = Avg_abs_rel + mask_abs_rel(pred, valid_gt_depth, valid_mask)
                    Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(pred, valid_gt_depth, 1, valid_mask)
                Avg_mse = Avg_mse / num_val
                Avg_abs_rel = Avg_abs_rel / num_val
                Avg_accuracy_1 = Avg_accuracy_1 / num_val

                print("Avg_mse(" + str(epoch) + ") : ", Avg_mse)
                print("Avg_abs_rel(" + str(epoch) + ") : ", Avg_abs_rel)
                print("Avg_accuracy_1(" + str(epoch) + ") : ", Avg_accuracy_1)

                writer.add_scalar("valid/Avg_mse", Avg_mse, epoch)
                writer.add_scalar("valid/Avg_abs_rel", Avg_abs_rel, epoch)
                writer.add_scalar("valid/Avg_accuracy_1", Avg_accuracy_1, epoch)
            # if Avg_mse <= Inf_loss:
            #     Inf_loss = Avg_mse
            #     path = record_root + "model_best.pth"
            #     torch.save(model.state_dict(), path)
            # print(Inf_loss)
            if Avg_abs_rel <= 0.1983 and Avg_mse <= 0.0720:
                path = record_root + "model_best_Avg_abs_rel.pth"
                torch.save(model.state_dict(), path)
                print("OK")

        # Training session
        model.train()
        total_loss_record = 0.0
        for idx, samples in enumerate(tqdm(train_dataloader, desc="Train")):  # check variable ranges, images
            train_input, train_gt_depth, train_focus_dists, train_mask, train_conf, _ = samples

            train_input = train_input.cuda(non_blocking=True)
            train_focus_dists = train_focus_dists.cuda(non_blocking=True)
            train_gt_depth = train_gt_depth.cuda(non_blocking=True)
            train_mask = train_mask.cuda(non_blocking=True)
            train_conf = train_conf.cuda(non_blocking=True)

            depth = model(train_input, train_focus_dists)

            depth = (depth - min_Depth) / (max_Depth - min_Depth)
            train_gt_depth = (train_gt_depth - min_Depth) / (max_Depth - min_Depth)

            optimizer.zero_grad()
            char_l = charbonnier_conf_loss(depth, train_gt_depth, train_conf, train_mask)
            edge_l = edge_conf_loss(depth, train_gt_depth, train_conf, train_mask)
            fft_l = fft_loss(depth, train_gt_depth)
            total_loss = c * char_l + e * edge_l + f * fft_l
            total_loss.backward()
            optimizer.step()
            total_loss_record = total_loss_record + total_loss.detach().data

        print("Epoch:", epoch, "AVG_TotalLoss:", total_loss_record / num_train)
        writer.add_scalar("train/Total_loss", total_loss_record / num_train, epoch)

    writer.close()


if __name__ == "__main__":
    main()
