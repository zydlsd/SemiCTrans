#! /usr/bin/env python
# -*- encoding: utf-8 -*-
# @File   :  unet2d.py
# @Time   :  2023/09/29 15:22:29
# @Author :  zyd


"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn


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

    # up_type=0: nn.ConvTranspose2d # 转置卷积 在input tensor的元素之间（横向、纵向）补0
    # up_type=1: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)双线性插值
    # up_type=2: nn.Upsample(scale_factor=2, mode='nearest')最近邻插值

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        # print('unet mode_upsampling={}'.format(mode_upsampling))

        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)

        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling == 1:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

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
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

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
        return output


# ----------------------------------------------------------------------------------------
# up_type=0: nn.ConvTranspose2d # 转置卷积 在input tensor的元素之间（横向、纵向）补0
# up_type=1: nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)双线性插值
# up_type=2: nn.Upsample(scale_factor=2, mode='nearest')最近邻插值
# scale_factor：指定输出为输入的多少倍数；
# ----------------------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_chns, class_num, up_type=0):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': up_type,
                   # 'up_type': 1,
                   # 'up_type': 2,
                   'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder(feature)
        return output1

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision import models

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    unet_model = UNet(in_chns=4, class_num=4, up_type=0)
    # 定义输入（假设输入大小为 (batch_size, channels, height, width)）
    input_tensor = torch.randn((1, 4, 224, 224))
    # 前向传播，获取倒数第二层的特征
    features = unet_model.encoder(input_tensor)
    output_features0 = unet_model.decoder.up1(features[4], features[3])
    output_features1 = unet_model.decoder.up2(output_features0, features[2])
    output_features2 = unet_model.decoder.up3(output_features1, features[1])
    output_features3 = unet_model.decoder.up4(output_features2, features[0])
    output_features4 = unet_model.decoder.out_conv(output_features3)
    # print("Feature Shape:", features.shape)
    # output_features = features[0]
    output_feat0= to_numpy(output_features0)
    output_feat1 = to_numpy(output_features1)
    output_feat2 = to_numpy(output_features2)
    output_feat3 = to_numpy(output_features3)
    output_feat4 = to_numpy(output_features4)

    # 打印某些层特征的形状
    print("output_feat0 Shape:", output_feat0.shape)
    print("output_feat1 Shape:", output_feat1.shape)
    print("output_feat2 Shape:", output_feat2.shape)
    print("output_feat3 Shape:", output_feat3.shape)
    print("output_feat4 Shape:", output_feat4.shape)

    """
    output_feat0 Shape: (1, 128, 28, 28)
    output_feat1 Shape: (1, 64, 56, 56)
    output_feat2 Shape: (1, 32, 112, 112)
    output_feat3 Shape: (1, 16, 224, 224) 
    output_feat4 Shape: (1, 4, 224, 224) 
    """

