import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class SFTLayer(nn.Module):
    def __init__(self, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, nf, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, nf, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.sft0 = SFTLayer(64)
        self.sft1 = SFTLayer(32)
        self.sft2 = SFTLayer(32)
        self.sft3 = SFTLayer(32)
        self.sft4 = SFTLayer(32)
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x0_sft = self.sft0((x[0], x[1]))
        x1 = self.lrelu(self.conv1(x0_sft))
        x1_sft = self.sft1((x1, x[1]))
        x2 = self.lrelu(self.conv2(torch.cat((x[0], x1_sft), 1)))
        x2_sft = self.sft2((x2, x[1]))
        x3 = self.lrelu(self.conv3(torch.cat((x[0], x1_sft, x2_sft), 1)))
        x3_sft = self.sft3((x3, x[1]))
        x4 = self.lrelu(self.conv4(torch.cat((x[0], x1_sft, x2_sft, x3_sft), 1)))
        x4_sft = self.sft4((x4, x[1]))
        x5 = self.conv5(torch.cat((x[0], x1_sft, x2_sft, x3_sft, x4_sft), 1))
        return x5 * 0.2 + x[0]

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1((x[0],x[1]))
        out = self.RDB2((out,x[1]))
        out = self.RDB3((out,x[1]))
        return (out * 0.2 + x[0], x[1])


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.CondNet = nn.Sequential(nn.Conv2d(1, 1, 1, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(1, 32, 1))

    def forward(self, x):
        cond = self.CondNet(x[1])
        fea = self.conv_first(x[0])
        fea2 = (fea, cond)
        fea3 = self.RRDB_trunk(fea2)
        trunk = self.trunk_conv(fea3[0])
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
