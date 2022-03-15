from ..layers.convolution import Conv2dBnRelu, Conv2dBn
from ..layers.blocks import BasicResNetBlock, BottleneckResNetBlock
from ..initialize import weight_initialize

import torch
from torch import nn

class ResNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetStem, self).__init__()
        self.conv = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3,
                        dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output

class _ResNet(nn.Module):
    def __init__(self, in_channels, block, layers, classes, dilated=True):
        super(_ResNet, self).__init__()
        self.in_channels = 64
        self.resnetStem = ResNetStem(in_channels=in_channels, out_channels=self.in_channels)
   
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        if dilated:
            self.layer3 = self.make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self.make_layer(block, 512, layers[3], stride=1, dilation=4)
        else:
            self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, classes) # convolution
        )
    
    def make_layer(self, block, channels, blocks, stride=1, dilation=1):
        downsample=None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = Conv2dBn(in_channels=self.in_channels, out_channels=channels * block.expansion,
                            kernel_size=1, stride=stride)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.in_channels, channels, stride, dilation=1, downsample=downsample, previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.in_channels, channels, stride, dilation=2, downsample=downsample, previous_dilation=dilation))
        else:
            raise RuntimeError("unknown dilation size : {}".format(dilation))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, dilation=dilation, previous_dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, input):
        stem= self.resnetStem(input)
        output = self.layer1(stem)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        print('shape : ', output.shape)
        pred = self.classifier(output)
        return {'pred': pred}


def ResNet(in_channels, classes=1000, varient=50):
    if varient == 18:
        model = _ResNet(in_channels=in_channels, block=BasicResNetBlock, layers=[2, 2, 2, 2], classes=classes)
    elif varient == 34:
        model = _ResNet(in_channels=in_channels, block=BasicResNetBlock, layers=[3, 4, 6, 3], classes=classes)
    elif varient == 50:
        model = _ResNet(in_channels=in_channels, block=BottleneckResNetBlock, layers=[3, 4, 6, 3], classes=classes)
    elif varient == 101:
        model = _ResNet(in_channels=in_channels, block=BottleneckResNetBlock, layers=[3, 4, 23, 3], classes=classes)
    elif varient == 152:
        model = _ResNet(in_channels=in_channels, block=BottleneckResNetBlock, layers=[3, 8, 36, 3], classes=classes)
    else:
        raise RuntimeError("unknown varient ResNet_{}".format(varient))
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = ResNet(in_channels=3, classes=1000, varient=50)
    model(torch.rand(1, 3, 320, 320))