#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
import argparse
import random
import numpy as np
from utils.net_utils.unet import UNet
from utils.net_utils.vision_transformer import SwinUnet
from utils.net_utils.configs.config import get_config
from utils.test_utils.tester import test_all_case


def test_calculate_metric(model, list_):
    test_all_case(net=model,
                  image_list=list_,
                  num_classes=args.num_classes,
                  patch_size=args.crop_size,
                  stride_xy=10,
                  lcc=args.lcc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=4, help='num_classes')
    parser.add_argument('--crop_size', type=int, default=(224, 224), help='patch_size')
    parser.add_argument('--net_in_chns', type=int, default=12)

    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/our_model_1percent_u.pth'
                        # default='./checkpoints/our_model_10percent_u.pth'
                        # default='./checkpoints/our_model_20percent_s.pth'
                        )

    parser.add_argument('--lcc', type=int, default=1, help='apply NMS post-procssing? 0/1,user-defined')
    parser.add_argument('--data_path', type=str,
                        default='./SemiCTrans/src/data/Brats2020Test/',

                        help='user-defined')
    parser.add_argument('--gpu', type=str, default='1', help='user-defined')

    # SwinUnet
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--cfg', type=str, default="./utils/net_utils/configs/swin_tiny_patch4_window7_224_lite.yaml", help='user-defined')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()
    config = get_config(args)
    # ----------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(args.data_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.data_path + item.replace('\n', '') for item in image_list]

    # load net
    net_name = args.model_path.split('/')[-1].split('.')[0].split('_')[-1]

    if net_name == 'u':
        net = UNet(in_chns=args.net_in_chns, class_num=args.num_classes, up_type=0).cuda()
        net.load_state_dict(torch.load(args.model_path)["state_dict"])
    elif net_name == 's':
        net = SwinUnet(config, img_size=224, num_classes=args.num_classes).cuda()
        net.load_state_dict(torch.load(args.model_path)["state_dict"])
    else:
        print('--net_name={} Error!'.format(net_name))
        net = None

    test_calculate_metric(net, image_list)

    print(args.model_path)
