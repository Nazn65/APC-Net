from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

import torch.nn.functional as F
from einops import rearrange
import math


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], 2,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        output = torch.sigmoid(output)  # 将输出压缩到0-1之间的概率值
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()
        self.is_corr = True
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        if self.is_corr:
            self.corr = Corr(nclass=class_num)
            self.proj = nn.Sequential(
                nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x, mask_x=None, need_fp=False, use_corr=False):
        dict_return = {}
        h, w = x.shape[-2:]
        feature = self.encoder(x)
        c4 = feature[-1]

        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            out, out_fp = outs.chunk(2)
            if use_corr:
                proj_feats = self.proj(c4)
                corr_out = self.corr(proj_feats, out, mask=mask_x)
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp
            return dict_return

        output = self.decoder(feature)
        if use_corr:
            proj_feats = self.proj(c4)
            corr_out = self.corr(proj_feats, output, mask=mask_x)
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = output

        return dict_return


class Corr(nn.Module):
    def __init__(self, nclass=4):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(32, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(32, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out, mask=None):
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)

        if mask is not None:

            mask = mask.clamp(0, self.nclass - 1)


            onthot = torch.nn.functional.one_hot(mask,num_classes=self.nclass)
            onthot = rearrange(onthot, 'n h w c -> n c h w')
            onthot = F.interpolate(onthot * 1.0, (h_in, w_in), mode='nearest')  # out: 24, 4, 16, 16


            out[:4, :, :] = out[:4, :, :] * 0.0 + onthot
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear',
                                align_corners=True)  # feature: 24, 32, 16, 16
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')  # 24, 4, 256
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')  # 24, 4, 256
        out_temp = rearrange(out, 'n c h w -> n c (h w)')  # 24, 4, 256
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)

        fina_out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)


        return fina_out


def norm(w):
    h = torch.sqrt(w.sum(-1))
    d_inv = 1.0 / (h)
    d_inv = torch.unsqueeze(d_inv, dim=-1) * torch.unsqueeze(d_inv, dim=-2)
    w = w * d_inv
    return w