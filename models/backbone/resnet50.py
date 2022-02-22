from ..layers.convolution import Conv2dBn, Conv2dBnAct
from ..layers.blocks import Residual_Block, Residual_LiteBlock
from ..initialize import weight_initialize

import torch
from torch import nn

class ResNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output


class Make_Layers(nn.Module):
    def __init__(self, layers_configs, pre_layer_ch):
        super(Make_Layers, self).__init__()
        self.pre_ch = pre_layer_ch
        self.layers_configs = layers_configs
        self.layer = self.residual_layer(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def residual_layer(self, cfg):
        layers = []
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            if i == 0:
                layer = Residual_Block(in_channels=self.pre_ch, kernel_size=cfg[1], out_channels=cfg[2], stride=cfg[3])
            else:
                layer = Residual_Block(in_channels=input_ch, kernel_size=cfg[1], out_channels=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channel()
        return nn.Sequential(*layers)
    
    
class Make_LiteLayer(nn.Module):
    def __init__(self, layers_configs, pre_layer_ch):
        super(Make_LiteLayer, self).__init__()
        self.pre_ch = pre_layer_ch
        self.layers_configs = layers_configs
        self.layer = self.residual_litelayer(self.layers_configs)
        
    def forward(self, input):
        return self.layer(input)
    
    def residual_litelayer(self, cfg):
        layers = []
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            if i == 0:
                layer = Residual_LiteBlock(in_channels=self.pre_ch, kernel_size=cfg[1], out_channels=cfg[2], stride=cfg[3])
            else:
                layer = Residual_LiteBlock(in_channels=input_ch, kernel_size=cfg[1], out_channels=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channel()
        return nn.Sequential(*layers)

class _ResNet50(nn.Module):
    def __init__(self, in_channels, classes):
        super(_ResNet50, self).__init__()
        self.resnetStem = ResNetStem(in_channels=in_channels, out_channels=64)

        # configs : in_channels, kernel_size, out_channels, stride, iter_cnt
        conv2_x = [64, 3, 256, 1, 3]
        conv3_x = [128, 3, 512, 2, 4]
        conv4_x = [256, 3, 1024, 2, 6]
        conv5_x = [512, 3, 2048, 2, 3]
   
        self.layer1 = Make_Layers(conv2_x, 64)
        self.layer2 = Make_Layers(conv3_x, 256)
        self.layer3 = Make_Layers(conv4_x, 512)
        self.layer4 = Make_Layers(conv5_x, 1024)
        self.Segmentation = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 4, classes)
        )
    
    def forward(self, input):
        stem= self.resnetStem(input)
        output = self.layer1(stem)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        pred = self.Segmentation(output)
        print('shape : ', pred.shape)
        return {'pred': pred}


def ResNet(in_channels, classes=1000):
    model = _ResNet50(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = ResNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))