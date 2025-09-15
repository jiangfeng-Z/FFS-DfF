"""
@author: zjf
@create time: 2024/11/22 10:30
@desc: wavelet approximation-aware residual network for single image deraining
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MSE_loss = nn.MSELoss()


def Masked_MSE_loss(est, gt, mask):
    out = MSE_loss(est[mask], gt[mask])
    return out


class Charbonnier_Loss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-8):
        super(Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y, mask):
        diff = x[mask].to('cuda:0') - y[mask].to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class Charbonnier_Conf_Loss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-8):
        super(Charbonnier_Conf_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y, conf, mask):
        diff = x[mask].to('cuda:0') - y[mask].to('cuda:0')
        loss = torch.sum(conf[mask] * torch.sqrt((diff * diff) + (self.eps * self.eps))) / torch.sum(conf[mask])
        return loss


class Edge_Loss(nn.Module):
    def __init__(self):
        super(Edge_Loss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = Charbonnier_Loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, mask):
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        mask = torch.unsqueeze(mask, 1)
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')), mask)
        return loss


class Edge_Conf_Loss(nn.Module):
    def __init__(self):
        super(Edge_Conf_Loss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = Charbonnier_Conf_Loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, conf, mask):
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 1)
        mask = torch.unsqueeze(mask, 1)
        conf = torch.unsqueeze(conf, 1)
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')), conf, mask)
        return loss


class FFT_Loss(nn.Module):
    def __init__(self):
        super(FFT_Loss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss
