import torch
from torch import nn

def getPadding(kernel_size, mode='same'):
    if mode == 'same':
        return (int((kernel_size - 1) / 2), (int((kernel_size - 1) / 2)))
    else:
        return 0

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                     dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super(Conv2dBnRelu, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                     dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super(Conv2dBn, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        output = self.conv(input)
        return self.bn(output)


class DepthwiseConv2dBnRelu(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1,
                    dilation=1, bias=False, padding_mode='zeros'):
        super(DepthwiseConv2dBnRelu, self).__init__()
        self.conv = Conv2dBnRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class DepthSepConv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthSepConv2dBnRelu, self).__init__()
        self.depthwise = DepthwiseConv2dBnRelu(in_channels=in_channels, kernel_size=kernel_size, stride=stride)
        self.pointwise = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                        dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        output = self.act(output)
        return output

class DepthSepConv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthSepConv2dBn, self).__init__()
        self.depthwise = DepthwiseConv2dBnRelu(in_channels=in_channels, kernel_size=kernel_size, stride=stride)
        self.pointwise = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                        dilation=1, groups=1, bias=False, padding_mode='zeros')

    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output
