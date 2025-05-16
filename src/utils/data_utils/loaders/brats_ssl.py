#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import os
from typing import Tuple
import itertools
from torch.utils.data.sampler import Sampler


class BratsDataset(Dataset):

    def __init__(self, base_dir, split, transform, labeled_num=None):

        self.data_dir = base_dir
        self.split = split
        self.transform = transform
        self.labeled_num = labeled_num
        self.sample_list = open(os.path.join(base_dir, self.split + '.list')).readlines()
        if self.labeled_num is not None:
            lb_num = np.ceil(self.labeled_num * len(self.sample_list))
            self.sample_list = self.sample_list[:int(lb_num)]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # load data
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name)

        h5f = h5py.File(data_path, 'r')
        image = h5f['preprocess_brain'][:]
        label = h5f['label'][:]
        brain_mask = h5f['mask_brain'][:]

        if self.transform:
            image_, label_ = self.transform((image, label, brain_mask))
            sample = {'preprocess_brain': image_, 'label': label_, 'case_name': self.sample_list[idx].strip('\n')}

        return sample


class ToTensor(object):

    def __call__(self, img_and_label: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, label, _ = img_and_label
        image.astype(np.float32)

        if label.dtype == str("uint16") or label.dtype == str("float64"):
            label = np.int16(label)

        return torch.Tensor(image.copy()), torch.Tensor(label.copy()).long()


class TwoStreamBatchSampler(Sampler):
    def __init__(self, labeled_idxs, unlabeled_idxs, batch_size, labeled_batch_size):
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        self.labeled_batch_size = labeled_batch_size

        assert len(self.labeled_idxs) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idxs) >= self.unlabeled_batch_size > 0

    def __iter__(self):
        unlabeled_iter = iterate_once(self.unlabeled_idxs)
        labeled_iter = iterate_eternally(self.labeled_idxs)

        return (labeled_batch + unlabeled_batch
                for (labeled_batch, unlabeled_batch) in zip(
            grouper(labeled_iter, self.labeled_batch_size),
            grouper(unlabeled_iter, self.unlabeled_batch_size)))

    def __len__(self):
        return len(self.unlabeled_idxs) // self.unlabeled_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
