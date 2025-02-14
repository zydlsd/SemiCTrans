#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division, print_function
import torch.nn as nn


class ProjectHead(nn.Module):
    def __init__(self, ndf=64, class_num=4):
        super(ProjectHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=class_num, out_channels=ndf, kernel_size=1)
        self.bn = nn.BatchNorm2d(ndf)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=1)

    def forward(self, outputs_logit):
        x = self.conv1(outputs_logit)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ClassContraNet(nn.Module):

    def __init__(self, class_num, ndf=64):
        super(ClassContraNet, self).__init__()
        self.conv0 = nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=7)

        self.fc1 = nn.Linear(ndf * 8, 512)
        self.fc2 = nn.Linear(512, class_num)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)

    def forward(self, seg_map):
        batch_size = seg_map.shape[0]
        map_feature = self.conv0(seg_map)

        x = self.leaky_relu(map_feature)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = self.avgpool2(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)

        x = self.fc2(x)

        return x
