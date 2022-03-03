import math
import torch
from torch import nn
from torch.nn import functional as F
from models.initialize import weight_initialize
from models.layers.convolution import Conv2dBn, Conv2dBnAct


class DeepLab(nn.Module):
    def __init__(self, Backbone, num_classes, in_channels=3):
        super(DeepLab, self).__init__()

        self.backbone = Backbone(in_channels, num_classes)
        self.high_level_channels = 2048
        self.low_level_channels = 256
        self.aspp_dilate = [6, 12, 18]

        self.aspp = ASPP(self.high_level_channels, self.aspp_dilate)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        weight_initialize(self)

    def forward(self, x):
        stem = self.backbone.resnetStem(x)
        s1 = self.backbone.layer1(stem)
        s2 = self.backbone.layer2(s1)
        s3 = self.backbone.layer3(s2)
        s4 = self.backbone.layer4(s3)

        output = self.aspp(s4)
        output = self.classifier(output)
        return output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules =[
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

if __name__ == '__main__':
    from models.backbone.resnet import ResNet
    model = DeepLab(
        Backbone=ResNet,
        num_classes=20,
        in_channels=3
    )
    print(model(torch.rand(1, 3, 320, 320)))