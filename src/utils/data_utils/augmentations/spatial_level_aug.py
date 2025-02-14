#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import cv2
import random
import numpy as np
from typing import Tuple


class RandomMirrorFlip(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:

        modalities, label, brain_mask = img_and_mask

        if random.random() < self.p:
            ax = np.random.randint(1, 3)
            modalities = np.flip(modalities, axis=ax)
            if label is not None:
                label = np.flip(label, axis=ax - 1)
            if brain_mask is not None:
                brain_mask = np.flip(brain_mask, axis=ax - 1)

        return modalities, label, brain_mask


class RandomRotation90(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    @staticmethod
    def _augment_rot90(sample_data, sample_seg, brain_mask=None, num_rot=(1, 2, 3), axes=(0, 1)):

        num_rot = np.random.choice(num_rot)
        axes = np.random.choice(axes, size=2, replace=False)
        axes.sort()
        axes_data = [i + 1 for i in axes]
        sample_data = np.rot90(sample_data, num_rot, axes_data)

        if sample_seg is not None:
            sample_seg = np.rot90(sample_seg, num_rot, axes)

        if brain_mask is not None:
            brain_mask = np.rot90(brain_mask, num_rot, axes)

        return sample_data, sample_seg, brain_mask

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:

        modalities, label, brain_mask = img_and_mask
        if random.random() < self.p:
            modalities, label, brain_mask = self._augment_rot90(modalities, label, brain_mask)
        return modalities, label, brain_mask
