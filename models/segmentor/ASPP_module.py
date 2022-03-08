import torch
from torch import nn
from torch.nn import functional as F
from models.layers.convolution import Conv2dBn, Conv2dBnRelu

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale):
        super(ASPP, self).__init__()
        '''
        output_stride = resolution / last_shape = (320 / 40) = 8
        8 : [12, 24, 36]
        16 : [6, 12, 18]
        '''
        self.branch1 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch2 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12, dilation=12,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch3 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=24, dilation=24,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch4 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=36, dilation=36,
                            groups=1, bias=True, padding_mode='zeros')
        # self.branch5_avg = nn.AvgPool2d(kernel_size=1)
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_up = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.branch5 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode='zeros')
    
        # resolution / last shape = 320 / 40 = 8(output_stride)

    def forward(self, input):
        input_h = input.size()[2] # (== h / 16)
        input_w = input.size()[3] # (== w / 16)
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        pre_branch5 = self.branch5_avg(input)
        branch5 = self.branch5(pre_branch5)
        branch5 = self.branch5_up(branch5)
        output = torch.cat([branch1, branch2, branch3, branch4, branch5], axis=1)
        return output

class Decoder(nn.Module): # using deeplabv3
    def __init__(self, out_channels, branch, num_classes):
        super(Decoder, self).__init__()
        self.conv = Conv2dBnRelu(in_channels=out_channels * branch, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                        groups=1, bias=True, padding_mode='zeros')
        self.out = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.out(output)
        return output