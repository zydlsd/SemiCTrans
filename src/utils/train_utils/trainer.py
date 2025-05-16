#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.train_utils import ramps
from utils.train_utils.losses import DiceLoss, mse_loss
from utils.data_utils.augmentations import spatial_level_aug, cropping
from utils.data_utils.loaders.brats_ssl import *


def calculated_weight_train_data(args):
    trainloader_ = data_load(args)
    label0_sum = 0
    label1_sum = 0
    label2_sum = 0
    label3_sum = 0

    for i_batch, sampled_batch in enumerate(trainloader_):
        volume_batch, label_batch = sampled_batch['preprocess_brain'], sampled_batch['label']
        lb_gt = label_batch[:]
        label0_pixel_count = (lb_gt == 0).sum()
        label1_pixel_count = (lb_gt == 1).sum()
        label2_pixel_count = (lb_gt == 2).sum()
        label3_pixel_count = (lb_gt == 3).sum()
        label0_sum += label0_pixel_count
        label1_sum += label1_pixel_count
        label2_sum += label2_pixel_count
        label3_sum += label3_pixel_count
    sum = label0_sum + label1_sum + label2_sum + label3_sum

    proportion_0 = label0_sum / sum
    proportion_1 = label1_sum / sum
    proportion_2 = label2_sum / sum
    proportion_3 = label3_sum / sum

    weight_0 = 1 / (args.num_classes * proportion_0)
    weight_1 = 1 / (args.num_classes * proportion_1)
    weight_2 = 1 / (args.num_classes * proportion_2)
    weight_3 = 1 / (args.num_classes * proportion_3)

    weight_0 = round(weight_0.item(), 2)
    weight_1 = round(weight_1.item(), 2)
    weight_2 = round(weight_2.item(), 2)
    weight_3 = round(weight_3.item(), 2)

    return [weight_0, weight_1, weight_2, weight_3]


def data_load(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BratsDataset(base_dir=args.data_path, split="train",
                            transform=transforms.Compose([
                                spatial_level_aug.RandomMirrorFlip(p=0.5),
                                spatial_level_aug.RandomRotation90(p=0.5),
                                cropping.RandomCrop(patch_size=args.crop_size),
                                ToTensor()
                            ]))
    # Brats2020
    lb_num_dict = {0.01: 30, 0.1: 300, 0.2: 600, 1: 3000}
    labeled_slices_num = lb_num_dict[args.label_rate]
    labeled_idxs = list(range(labeled_slices_num))
    unlabeled_idxs = list(range(labeled_slices_num, len(db_train)))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    return trainloader


def train_sup_loss(args):
    if args.ce_weight is not None:
        criterion_ce = CrossEntropyLoss(weight=torch.tensor(args.ce_weight).cuda())
    else:
        criterion_ce = CrossEntropyLoss()

    if args.dice_weight is not None:
        criterion_dice = DiceLoss(args.num_classes, weight=args.dice_weight)
    else:
        criterion_dice = DiceLoss(args.num_classes, weight=None)

    return criterion_dice, criterion_ce


def DUGM(data, outputs, model, args, T, now_epoch):
    net_num = 2
    stride = data.shape[0]
    outputs_soft1 = torch.softmax(outputs[0], dim=1)
    outputs_soft2 = torch.softmax(outputs[1], dim=1)
    uncertainty = []
    for i in range(net_num):
        pred_bayes = torch.zeros([stride * T, args.num_classes, args.crop_size[0], args.crop_size[1]]).cuda()
        for t in range(T):
            noise_data = data + torch.clamp(torch.rand_like(data) * 0.1, -0.2, 0.2).cuda()
            with torch.no_grad():
                pred_bayes[stride * t:  stride * (t + 1)] = model[i](noise_data)
        pred_bayes = torch.softmax(pred_bayes, dim=1)
        pred_bayes = pred_bayes.reshape(T, stride, args.num_classes, args.crop_size[0], args.crop_size[1])
        pred_bayes = torch.mean(pred_bayes, dim=0)
        uncert = -1.0 * torch.sum(pred_bayes * torch.log(pred_bayes + 1e-6), dim=1, keepdim=True)
        uncertainty.append(uncert)

    consistency_dist = mse_loss(outputs_soft1, outputs_soft2)
    threshold = 0.75 + 0.25 * ramps.sigmoid_rampup(now_epoch, args.consistency_rampup)
    mask = ((uncertainty[0] < threshold) & (uncertainty[1] < threshold)).float()
    consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

    top = (outputs_soft1 / uncertainty[0] + outputs_soft2 / uncertainty[1]).cuda()
    down = (1 / uncertainty[0] + 1 / uncertainty[1]).cuda()
    pseudo = top / down
    pseudo_label = torch.argmax(pseudo, dim=1)

    return consistency_loss, pseudo_label
