import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input):
        pred, target = input
        ce_loss = F.cross_entropy(
            pred, target, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

    # def forward(self, input):
    #     total_loss = []
    #     pred, target = input
    #     pred = torch.clamp(pred, 1e-7, 1.)
    #     batch_size = pred.shape[0]

    #     for b in range(batch_size):
    #         loss = self.loss_function(pred[b], target[b])
    #         loss = loss.sum()
    #         total_loss.append(loss)
        
    #     return torch.stack(total_loss).mean()
        



    # def loss_function(self, pred, target):
    #     alpha_factor = torch.where(
    #         torch.eq(target, 1.), self.alpha, 1.-self.alpha)
    #     focal_weight = torch.where(torch.eq(target, 1.), 1.-pred, pred)
    #     focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
    #     binary_ce = -(target * torch.log(pred) +
    #                   (1.-target)*torch.log(1.-pred))
    #     loss = focal_weight*binary_ce
    #     return loss