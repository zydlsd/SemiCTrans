#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import h5py
import psutil
import math
import torch

from tqdm import tqdm
from skimage.measure import label
from utils.test_utils.evaluation import *


def getLargestCC(segmentation):
    # Return a mask corresponding to the largest object
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    else:
        assert (labels.max() != 0)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC


def compute(pred, gt, brain_mask):
    # http://loli.github.io/medpy/metric.html
    if len(np.unique(pred)) == 1 and len(np.unique(gt)) == 1 and np.unique(pred)[0] == 0 and np.unique(gt)[0] == 0:
        dc_ = 1.0
        jc_ = 1.0
        hd_ = 0.0
        asd_ = 0.0
    else:
        tp, fp, tn, fn = get_confusion_matrix(pred, gt, brain_mask)
        dc_ = dice(tp, fp, fn)
        jc_ = jc(tp, fp, fn)
        hd_ = hausdorff(pred, gt)
        asd_ = asd(pred, gt)

    return dc_, jc_, hd_, asd_


def compute_wt_tc_et(pred, gt, brain_mask):
    WT = []
    TC = []
    ET = []
    for typee in ["wt", "tc", "et"]:
        if typee == "wt":
            volume_pred = (pred > 0)
            volume_gt = (gt > 0)
            WT.append(compute(volume_pred, volume_gt, brain_mask))

        elif typee == "tc":
            volume_pred = (pred == 1) | (pred == 3)
            volume_gt = (gt == 1) | (gt == 3)
            TC.append(compute(volume_pred, volume_gt, brain_mask))

        elif typee == "et":
            volume_pred = (pred == 3)
            volume_gt = (gt == 3)
            if len(np.unique(volume_gt)) == 1 and np.unique(volume_gt)[0] == 0:
                print("no Enchancing Tumor in GT!")
            ET.append(compute(volume_pred, volume_gt, brain_mask))

        else:
            continue

    return WT[0], TC[0], ET[0]


def test_all_case(net, image_list, num_classes, patch_size, stride_xy, lcc=0):

    total_metric_wt = []
    total_metric_tc = []
    total_metric_et = []

    ith = 1
    net.eval()
    for image_path in tqdm(image_list, ncols=70):
        h5f = h5py.File(image_path, 'r')
        img = h5f['preprocess_brain'][:]
        lb = h5f['label'][:]
        brain_mask = h5f['mask_brain'][:]

        # =============================================================
        prediction = np.zeros(lb.shape).astype(np.float32)
        layers = lb.shape[0]

        for layer in range(layers):
            if layer == 0:
                layer_indices = [0, 0, 1]
            elif layer == layers - 1:
                layer_indices = [layers - 2, layers - 1, layers - 1]
            else:
                layer_indices = [layer - 1, layer, layer + 1]

            no_stacked_data = img[:, layer_indices]
            c, lys, h, w = no_stacked_data.shape
            stacked_data = np.reshape(no_stacked_data, (c * lys, h, w))
            assert stacked_data.shape == (c * lys, h, w)

            pred, _ = test_case(net, stacked_data, stride_xy, patch_size, classes=num_classes)
            prediction[layer] = pred

        assert prediction.shape == lb.shape

        if lcc:
            prediction_ = prediction.copy()
            prediction_[prediction_ == 2] = 1
            prediction_[prediction_ == 3] = 1
            prediction_mask = getLargestCC(prediction_)
            prediction = prediction_mask * prediction

        single_metric = compute_wt_tc_et(prediction, lb, brain_mask)

        print('\ncase%01d[WT]:  dice:%.2f  jc:%.2f  hd95:%.2f  asd:%.2f' % (
            ith, single_metric[0][0], single_metric[0][1], single_metric[0][2], single_metric[0][3]))
        print('case%01d[TC]:  dice:%.2f  jc:%.2f  hd95:%.2f  asd:%.2f' % (
            ith, single_metric[1][0], single_metric[1][1], single_metric[1][2], single_metric[1][3]))
        print('case%01d[ET]:  dice:%.2f  jc:%.2f  hd95:%.2f  asd:%.2f\n' % (
            ith, single_metric[2][0], single_metric[2][1], single_metric[2][2], single_metric[2][3]))

        total_metric_wt.append(single_metric[0])
        total_metric_tc.append(single_metric[1])
        total_metric_et.append(single_metric[2])

        ith += 1

    mean_metric_wt = np.mean(np.array(total_metric_wt), axis=0)
    mean_metric_tc = np.mean(np.array(total_metric_tc), axis=0)
    mean_metric_et = np.mean(np.array(total_metric_et), axis=0)

    print('=='*25)
    print('Average score for all test data:'
          '\n     dice,  jc,  hd95, asd'
          '\nWT %.4f %.4f %.2f %.2f'
          '\nTC %.4f %.4f %.2f %.2f'
          '\nET %.4f %.4f %.2f %.2f' %
          (mean_metric_wt[0], mean_metric_wt[1], mean_metric_wt[2], mean_metric_wt[3],
           mean_metric_tc[0], mean_metric_tc[1], mean_metric_tc[2], mean_metric_tc[3],
           mean_metric_et[0], mean_metric_et[1], mean_metric_et[2], mean_metric_et[3]))
    print('==' * 25)


def test_case(net, image, stride_xy, patch_size, classes):
    channel, w, h = image.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0

    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2

    if add_pad:
        image = np.pad(image, [(0, 0), (wl_pad, wr_pad), (hl_pad, hr_pad)], mode='constant', constant_values=0)

    ch, ww, hh = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1

    score_map = np.zeros((classes,) + (ww, hh)).astype(np.float32)
    cnt = np.zeros((ww, hh)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1]]
            assert test_patch.shape == (channel, patch_size[0], patch_size[1])
            test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
            test_patch = torch.from_numpy(test_patch).cuda()
            with torch.no_grad():
                y1 = net(test_patch)
                y = torch.softmax(y1, dim=1)
            y = y.cpu().data.numpy()
            y = y[0, :, :, :]
            score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1]] \
                = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1]] + y
            cnt[xs:xs + patch_size[0], ys:ys + patch_size[1]] \
                = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1]] + 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h]

    return label_map, score_map
