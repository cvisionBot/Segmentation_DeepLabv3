import torch
from torch import nn
from torch.nn import functional as F
from models.layers.convolution import Conv2dBn, Conv2dBnRelu

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ASPP, self).__init__()

        self.branch1 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch2 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6, dilation=6,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch3 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=12, dilation=12,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch4 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=18, dilation=18,
                            groups=1, bias=True, padding_mode='zeros')
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5 = Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, dilation=1,
                            groups=1, bias=True, padding_mode='zeros')
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1), # 1280 input = 5 * 256
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
        )
    
    def forward(self, input):
        input_h = input.size()[2] # (== h / 16)
        input_w = input.size()[3] # (== w / 16)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        pre_branch5 = self.branch5_avg(input)
        branch5 = self.branch5(pre_branch5)

        out_img = F.upsam
