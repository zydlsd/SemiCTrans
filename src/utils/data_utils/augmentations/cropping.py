#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Tuple
import numpy as np


class RandomCrop(object):
    def __init__(self, patch_size=128):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, img_and_mask: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        modalities, label, brain_mask = img_and_mask  #
        # pad the sample if necessary
        if label.shape[0] <= self.patch_size[0] or label.shape[1] <= self.patch_size[1]:
            pw = max((self.patch_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.patch_size[1] - label.shape[1]) // 2 + 3, 0)
            modalities = np.pad(modalities, [(0, 0), (pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (_, w, h) = modalities.shape
        w1 = np.random.randint(0, w - self.patch_size[0])
        h1 = np.random.randint(0, h - self.patch_size[1])

        image_patch = modalities[:, w1:w1 + self.patch_size[0], h1:h1 + self.patch_size[1]]
        label_patch = label[w1:w1 + self.patch_size[0], h1:h1 + self.patch_size[1]]

        assert image_patch[0].shape == label_patch.shape == self.patch_size

        return image_patch, label_patch, brain_mask
