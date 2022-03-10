import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from utils.module_select import get_optimizer
from models.loss.focal_loss import FocalLoss

class Segmentor(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super(Segmentor, self).__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = FocalLoss()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        loss = self.opt_training_step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['img'])
        loss = self.loss_fn([pred, batch['label']])
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def opt_training_step(self, batch):
        pred = self.model(batch['img'])
        loss = self.loss_fn([pred, batch['label']])
        return loss


    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(cfg['optimizer'],
            params=self.model.parameters(),
            **cfg['optimizer_options']
        )
        return optim