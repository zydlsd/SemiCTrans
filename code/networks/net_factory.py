#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :  net_factory.py
@Time   :  2023/09/29 15:03:50
@Author :  zyd
'''
# 2d
from DCSS.src.utils.net_utils.unet2d import UNet
from DCSS.src.utils.net_utils.class_contra_net import ClassContra2DNet
# 3d
from DCSS.src.utils.net_utils.vnet import VNet
from DCSS.src.utils.net_utils.unet3d import UNet3D


def net_factory_2d(net_type, in_chns=4, class_num=4, up_type=0):
    
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num, up_type=up_type)

    elif net_type == "class_contra_net":
        # output: [class_vector, project_head for Contrastive Learning]
        net = ClassContra2DNet(class_num=class_num)
        
    else:
        net = "Error!"
    print("-----success load {} 2d network!-----".format(net_type))
    return net


def net_factory_3d(net_type, in_chns=4, class_num=4):

    if net_type == "unet3d":
        net = UNet3D(in_channels=in_chns, n_classes=class_num)

    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num, has_dropout=True, has_residual=False)

    else:
        net = "Error!"
    print("-----success load {} 3d network!-----".format(net_type))
    return net
