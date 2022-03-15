import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input):
        # input shape : ([pred, batch])
        pred, target = input
        # pred shape : (batch, classes, height, width)
        # target shape : (batch, classes, height, width)
        batch_size = pred.shape[0]
        class_size = pred.shape[1]

        total_loss = []
        for b in range(batch_size):
            batch_loss = []
            for c in range(class_size):
                ce_loss = F.cross_entropy(
                    pred[b, c, :, :], target[b, c, :, :]
                )
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                batch_loss.append(focal_loss)
            batch_loss = torch.stack(batch_loss).sum()
            total_loss.append(batch_loss)

        return torch.stack(total_loss).mean()