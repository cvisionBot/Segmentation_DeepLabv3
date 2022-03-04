import math
import torch
from torch import nn
from models.initialize import weight_initialize
from models.segmentor.ASPP_module import ASPP, Decoder


class DeepLab(nn.Module):
    def __init__(self, Backbone, num_classes, in_channels=3):
        super(DeepLab, self).__init__()
        self.h_res_channels = 2048
        self.backbone = Backbone(in_channels=in_channels, classes=num_classes)
        self.aspp = ASPP(in_channels= self.h_res_channels, out_channels= 256, num_classes=num_classes)
        self.decoder = Decoder(out_channels=256, up_stride=8, num_classes=num_classes)

    def forward(self, x):
        stem = self.backbone.resnetStem(x)
        s1 = self.backbone.layer1(stem)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)
        print('backbone shape : ', s4.shape)
        neck = self.aspp(s4)
        print('neck shape : ', neck.shape)
        output = self.decoder(neck)
        print('head shape : ', output.shape())
        return output




if __name__ == '__main__':
    from models.backbone.resnet import ResNet
    model = DeepLab(
        Backbone=ResNet,
        num_classes=20,
        in_channels=3,
    )
    weight_initialize(model)
    print(model(torch.rand(1, 3, 320, 320)))