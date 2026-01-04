import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.losses.loss_util import weighted_loss


_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean',): # loss_weight-损失权重，可以用于与其他损失函数结合时调整其比重;
        # reduction='mean'，对批量数据（batch）计算均值; toY=False，损失会基于 YUV 色彩空间中的 Y 通道（亮度通道）计算，更符合视觉
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)       #用于将 MSE 转换为 PSNR 的系数

    def forward(self, pred, target):
        assert len(pred.size()) == 4  #确保输入的 pred (预测图像)和 target（目标图像） 是 4 维张量，（b,c,h,w）
        # 计算PSNR损失，(pred - target) ** 2 是算预测和目标之间的平方差, .mean(dim=(1, 2, 3)) 是把每个图像的像素平均一下，得到每个样本的 MSE
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class MultiHeadPSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MultiHeadPSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.step = 0

    def forward(self, pred, target):
        assert len(pred.size()) == 5

        l2 = ((pred - target.unsqueeze(1)) ** 2).mean(dim=(2,3,4)) # N, H
        l2 = l2.min(dim=1)[0]
        return self.loss_weight * self.scale * torch.log(l2 + 1e-8).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.step = 0

    def forward(self, pred, target):
        assert len(pred.size()) == 5
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        return self.loss_weight * F.cross_entropy(pred, target)