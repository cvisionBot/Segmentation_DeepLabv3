import torch
import torch.nn.functional as F
from torch import nn

# next alpha 0.5
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input):
        # input shape : ([pred, batch])
        pred, target = input
        # pred shape : (batch, classes, height, width)
        # target shape : (batch, classes, height, width)
        pred_soft = F.softmax(pred, dim=1) + 1e-8
        weight = torch.pow(1.0 - pred_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(pred_soft)
        loss_tmp = torch.sum(target * focal, dim=1)
        loss = torch.mean(loss_tmp)
        return loss
