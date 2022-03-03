import torch
from torch import nn
from ..layers.convolution import Conv2dBnRelu, Conv2dBn

class BasicResNetBlock(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, previous_dilation=1, expansion=1):
        super(BasicResNetBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.expansion = expansion
        self.convbnrelu = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=dilation, dilation=dilation, bias=False)
        self.convbn = Conv2dBn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        output = self.convbnrelu(input)
        output = self.convbn(output)

        if self.downsample is not None:
            residual = self.downsample(input)
        output += residual
        output = self.relu(output)
        return output

class BottleneckResNetBlock(nn.Module):
    expansion=4
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, previous_dilation=1, expansion=4):
        super(BottleneckResNetBlock, self).__init__()
        self.stride=stride
        self.downsample=downsample
        self.dilation = dilation
        self.expansion = expansion
        self.convbnrelu1 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.convbnrelu2 = Conv2dBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
                                dilation=dilation, bias=False)
        self.convbn = Conv2dBn(in_channels=out_channels, out_channels=out_channels * expansion, kernel_size=1, stride=1, padding=0,
                                dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        output = self.convbnrelu1(input)
        output = self.convbnrelu2(output)
        output = self.convbn(output)

        if self.downsample is not None:
            residual = self.downsample(input)
        output += residual
        output = self.relu(output)
        return output