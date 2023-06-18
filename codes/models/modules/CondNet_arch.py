import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
import models.modules.sft_arch as sft
from .unet_parts import *

class CondNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(CondNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    #
    # def __init__(self, in_nc=52, out_nc=1, nf=64, nb=16, upscale=4):
    #     super(CondNet, self).__init__()
    #
    #     basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
    #     self.recon_trunk = mutil.make_layer(basic_block, nb)
    #
    #     self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
    #     # self.MaxPool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
    #     # self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
    #     self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
    #
    #     # activation function
    #     self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    #
    #     # initialization
    #     # mutil.initialize_weights([self.conv_first, self.upconv1, self.conv_last], 0.1)
    #     mutil.initialize_weights([self.conv_first, self.conv_last], 0.1)
    #
    #     # self.CondNet = nn.Sequential(nn.Conv2d(1, 1, 1, 1), nn.LeakyReLU(0.1, True),
    #     #                              # nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
    #     #                              # nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
    #     #                              # nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
    #     #                              nn.Conv2d(1, 64, 1))
    #
    # def forward(self, x):
    #     fea = self.conv_first(x)
    #     fea = self.lrelu(fea)
    #     fea = self.recon_trunk(fea)
    #     # out = self.MaxPool(out)
    #     # out = self.MaxPool(out)
    #     # fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='bicubic'))
    #     out = self.conv_last(fea)
    #     return out
    #     #
    #     # # x[0]: img; x[1]: seg
    #     # cond = self.CondNet(x[1])
    #     # fea = self.conv0(x[0])
    #     # res = self.sft_branch((fea, cond))
    #     # fea = fea + res
    #     # out = self.HR_branch(fea)
    #     # return out
