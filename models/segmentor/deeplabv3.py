import math
import torch
from torch import nn
from models.layers.convolution import Conv2dBn
from models.initialize import weight_initialize
from models.segmentor.ASPP_module import ASPP, Decoder


class DeepLab(nn.Module):
    def __init__(self, Backbone, num_classes, in_channels=3):
        super(DeepLab, self).__init__()
        self.backbone_output = 2048
        self.neck_input = 512
        self.last_layer = 40

        self.backbone = Backbone(in_channels=in_channels, classes=num_classes)
        self.pre_aspp = Conv2dBn(in_channels=self.backbone_output, out_channels=self.neck_input, kernel_size=1, stride=1, padding=0,
                            dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.aspp = ASPP(in_channels= self.neck_input, out_channels= 256, up_scale=self.last_layer)
        self.decoder = Decoder(out_channels=256, branch=5, num_classes=num_classes)

    def forward(self, x):
        stem = self.backbone.resnetStem(x)
        s1 = self.backbone.layer1(stem)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)
        neck = self.pre_aspp(s4)
        neck = self.aspp(neck)
        output = self.decoder(neck)
        return output




if __name__ == '__main__':
    from models.backbone.resnet import ResNet
    model = DeepLab(
        Backbone=ResNet,
        num_classes=20,
        in_channels=3,
    )
    weight_initialize(model)
    print(model(torch.rand(10, 3, 320, 320)))