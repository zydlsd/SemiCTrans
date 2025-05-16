#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CosineSimilarity


def process_features(feature, mask, mask_value):
    index = np.where(mask == mask_value)
    if len(index[0]) == 0:
        return None, None

    replace_id = len(index[0]) < 500
    selected_ids = np.random.choice(len(index[0]), size=500, replace=replace_id)
    selected_features = feature[index[0][selected_ids], :, index[1][selected_ids], index[2][selected_ids]]
    selected_features = selected_features.reshape(-1, feature.shape[1])
    return torch.from_numpy(selected_features).cuda(), selected_features


def class_feature_vectors(feature, mask):
    with torch.no_grad():
        feature = feature.cpu().numpy()
        mask = mask.cpu().numpy()

        selected_feature_vector_1, _ = process_features(feature, mask, 1)
        selected_feature_vector_2, _ = process_features(feature, mask, 2)
        selected_feature_vector_3, _ = process_features(feature, mask, 3)

        if selected_feature_vector_1 is not None and selected_feature_vector_2 is not None and selected_feature_vector_3 is not None:
            return selected_feature_vector_1, selected_feature_vector_2, selected_feature_vector_3
        elif selected_feature_vector_1 is None and selected_feature_vector_2 is not None and selected_feature_vector_3 is not None:
            return None, selected_feature_vector_2, selected_feature_vector_3
        elif selected_feature_vector_1 is not None and selected_feature_vector_2 is None and selected_feature_vector_3 is not None:
            return selected_feature_vector_1, None, selected_feature_vector_3
        elif selected_feature_vector_1 is not None and selected_feature_vector_2 is not None and selected_feature_vector_3 is None:
            return selected_feature_vector_1, selected_feature_vector_2, None
        else:
            return None, None, None


def compute_contrast_loss(class_1, class_2, class_3, ulb_class_1, ulb_class_2, ulb_class_3, contrast_temperature):
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    loss = 0

    if class_1 is not None and ulb_class_1 is not None:
        intra_cl1 = torch.exp((cos_sim(class_1, ulb_class_1)) / contrast_temperature)
        loss += -torch.log(intra_cl1) + torch.log(intra_cl1 + torch.exp((cos_sim(class_1, ulb_class_2)) / contrast_temperature) + torch.exp((cos_sim(class_1, ulb_class_3)) / contrast_temperature))

    if class_2 is not None and ulb_class_2 is not None:
        intra_cl2 = torch.exp((cos_sim(class_2, ulb_class_2)) / contrast_temperature)
        loss += -torch.log(intra_cl2) + torch.log(intra_cl2 + torch.exp((cos_sim(class_2, ulb_class_1)) / contrast_temperature) + torch.exp((cos_sim(class_2, ulb_class_3)) / contrast_temperature))

    if class_3 is not None and ulb_class_3 is not None:
        intra_cl3 = torch.exp((cos_sim(class_3, ulb_class_3)) / contrast_temperature)
        loss += -torch.log(intra_cl3) + torch.log(intra_cl3 + torch.exp((cos_sim(class_3, ulb_class_1)) / contrast_temperature) + torch.exp((cos_sim(class_3, ulb_class_2)) / contrast_temperature))

    return loss


def contrast_loss(args, feature1, feature2, gt, pseudo_gt):
    lb_feature_net12 = torch.cat((feature1[:args.labeled_bs], feature2[:args.labeled_bs]), dim=0)
    lb_gt_net12 = torch.cat((gt, gt), dim=0)

    ulb_feature_net12 = torch.cat((feature1[args.labeled_bs:], feature2[args.labeled_bs:]), dim=0)
    ulb_gt_net12 = torch.cat((pseudo_gt, pseudo_gt), dim=0)

    lb_class_1, lb_class_2, lb_class_3 = class_feature_vectors(lb_feature_net12, lb_gt_net12)
    ulb_class_1, ulb_class_2, ulb_class_3 = class_feature_vectors(ulb_feature_net12, ulb_gt_net12)

    if lb_class_1 is not None and ulb_class_1 is not None:
        if lb_class_2 is not None and ulb_class_2 is not None and lb_class_3 is not None and ulb_class_3 is not None:
            return torch.mean(compute_contrast_loss(lb_class_1, lb_class_2, lb_class_3, ulb_class_1, ulb_class_2, ulb_class_3, args.contrast_temperature))

        elif lb_class_2 is not None and ulb_class_2 is not None:
            return torch.mean(compute_contrast_loss(lb_class_1, lb_class_2, None, ulb_class_1, ulb_class_2, None, args.contrast_temperature))

        elif lb_class_3 is not None and ulb_class_3 is not None:
            return torch.mean(compute_contrast_loss(lb_class_1, None, lb_class_3, ulb_class_1, None, ulb_class_3, args.contrast_temperature))

    elif lb_class_2 is not None and ulb_class_2 is not None:
        if lb_class_3 is not None and ulb_class_3 is not None:
            return torch.mean(compute_contrast_loss(None, lb_class_2, lb_class_3, None, ulb_class_2, ulb_class_3, args.contrast_temperature))

    return 0


def mse_loss(input1, input2):
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        batch, x, y = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
        temp_prob = torch.zeros((batch, self.n_classes, x, y))
        for i in range(self.n_classes):
            temp_prob[:, i, :, :] = input_tensor == i * torch.ones_like(input_tensor)
        return temp_prob.float()

    @staticmethod
    def _dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target).cuda()
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * self.weight[i]
        return loss / self.n_classes
