"""
@author: zjf
@create time: 2024/2/25 17:07
@desc:
"""
import torch.nn as nn
from Tool.metrics import *
from Dataloader.Dataloader_FlyingThings3D import FlyingThings3D
from Model.Network import FFS_T
from Tool.Loss_Function import Charbonnier_Loss, Edge_Loss, FFT_Loss
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from torch.autograd import Variable
from Tool.min_norm_solvers import MinNormSolver
import torch.nn.functional as F


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"


charbonnier_loss = Charbonnier_Loss()
edge_loss = Edge_Loss()
fft_loss = FFT_Loss()

# loss 比例关系
c = 1.0
e = 0.001
f = 0.001

def sample_index(sample, interval):
    # 多景深图像序列数
    temp_count = []
    for i in range(0, sample, interval):
        temp_count.append(i)
    step = len(temp_count)
    index = torch.linspace(0, sample - 1, steps=step)
    index = torch.round(index).to(dtype=torch.long)
    return interval, index

def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--record_root', default="Record/FlyingThings3D_15_S/", type=str, help='save root')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=5000, type=int, help='max epoch')
    parser.add_argument('--load_epoch', default=0, type=int, help='load epoch')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')

    args = parser.parse_args()
    record_root = args.record_root
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    load_epoch = args.load_epoch
    # 参数设置
    writer = SummaryWriter(log_dir=record_root + "logs")

    max_Depth = 100.0
    min_Depth = 10.0
    num_train = 1000
    num_val = 100
    input_size = (540, 960)

    # 是采样多少  相当于总数是多少
    sample = 15
    train_1_interval, train_1_index = sample_index(sample, 2)
    train_2_interval, train_2_index = sample_index(sample, 4)
    val_interval, val_index = sample_index(sample, 3)

    Inf_loss = float('inf')
    model = FFS_T()
    train_dataset = FlyingThings3D('train')
    valid_dataset = FlyingThings3D('val')

    model = nn.DataParallel(model)
    if load_epoch:
        path = record_root + "model_best.pth"
        model.load_state_dict(torch.load(path))
    else:
        path = "Record/HCI_10_i_1_3/model_best.pth"
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
        if (epoch - load_epoch) >= 500:
            model.eval()
            with torch.no_grad():
                Avg_mse = 0.0
                Avg_abs_rel = 0.0
                Avg_accuracy_1 = 0.0
                for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
                    valid_input, valid_gt_depth, valid_focus_dists, valid_mask = samples
                    valid_input = valid_input[:, :, ::val_interval, :, :].cuda()
                    valid_focus_dists = valid_focus_dists[:, val_index, ...].cuda()
                    valid_gt_depth = np.squeeze(valid_gt_depth.numpy())
                    valid_mask = np.squeeze(valid_mask.data.cpu().numpy())
                    pred = model(valid_input, valid_focus_dists)
                    pred = np.squeeze(pred.data.cpu().numpy())
                    pred = pred[:input_size[0], :input_size[1]]
                    # print(pred.shape)
                    Avg_mse = Avg_mse + mask_mse(pred, valid_gt_depth, valid_mask)
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
            if Avg_mse <= Inf_loss:
                Inf_loss = Avg_mse
                path = record_root + "model_best.pth"
                torch.save(model.state_dict(), path)
            print(Inf_loss)

        # Training session
        model.train()

        for idx, samples in enumerate(tqdm(train_dataloader, desc="Train")):
            # 公共的
            train_input, train_gt_depth, train_focus_dists, train_mask = samples
            train_gt_depth = train_gt_depth.cuda(non_blocking=True)
            train_mask = train_mask.cuda(non_blocking=True)
            train_gt_depth = (train_gt_depth - min_Depth) / (max_Depth - min_Depth)

            # train 1
            train_input_1 = train_input[:, :, ::train_1_interval, :, :].cuda(non_blocking=True)
            train_focus_dists_1 = train_focus_dists[:, train_1_index, ...].cuda(non_blocking=True)
            depth_1 = model(train_input_1, train_focus_dists_1)
            depth_1 = (depth_1 - min_Depth) / (max_Depth - min_Depth)
            char_l = charbonnier_loss(depth_1, train_gt_depth, train_mask)
            edge_l = edge_loss(depth_1, train_gt_depth, train_mask)
            fft_l = fft_loss(depth_1, train_gt_depth)
            total_1_loss = c * char_l + e * edge_l + f * fft_l

            # train 2
            train_input_2 = train_input[:, :, ::train_2_interval, :, :].cuda(non_blocking=True)
            train_focus_dists_2 = train_focus_dists[:, train_2_index, ...].cuda(non_blocking=True)
            depth_2 = model(train_input_2, train_focus_dists_2)
            depth_2 = (depth_2 - min_Depth) / (max_Depth - min_Depth)
            char_l = charbonnier_loss(depth_2, train_gt_depth, train_mask)
            edge_l = edge_loss(depth_2, train_gt_depth, train_mask)
            fft_l = fft_loss(depth_2, train_gt_depth)
            total_2_loss = c * char_l + e * edge_l + f * fft_l

            losses = [total_1_loss, total_2_loss]
            losses_name = ['dense', 'sparse']
            grads_dense = {'dense': []}
            grads_sparse = {'sparse': []}
            record_dense = []
            record_sparse = []
            # 计算每个损失的梯度
            for idx, loss_type in enumerate(losses_name):
                loss = losses[idx]
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                if loss_type == 'dense':
                    for param in model.parameters():
                        grads_dense[loss_type].append(Variable(param.grad.data.clone(), requires_grad=False))
                        record_dense.append(Variable(param.data.clone().flatten(), requires_grad=False))
                elif loss_type == 'sparse':
                    for param in model.parameters():
                        grads_sparse[loss_type].append(Variable(param.grad.data.clone(), requires_grad=False))
                        record_sparse.append(Variable(param.data.clone().flatten(), requires_grad=False))

            record_dense = torch.cat(record_dense)
            record_sparse = torch.cat(record_sparse)
            this_cos_value = F.cosine_similarity(record_dense, record_sparse, dim=0)

            if this_cos_value > 0:
                multi_k = [0.5, 0.5]
            else:
                multi_k, min_norm = MinNormSolver.find_min_norm_element(
                    [torch.tensor(record_dense), torch.tensor(record_sparse)])
            optimizer.zero_grad()

            for idx, param in enumerate(model.parameters()):
                new_grad = multi_k[0] * grads_dense['dense'][idx] + multi_k[1] * grads_sparse['sparse'][idx]
                param.grad = new_grad
            optimizer.step()

    writer.close()


if __name__ == "__main__":
    main()
