#!/usr/bin/env python
# -*- coding:utf-8 -*-


from typing import Tuple
from medpy import metric
import numpy as np


def get_confusion_matrix(prediction: np.ndarray, reference: np.ndarray, roi_mask: np.ndarray) -> Tuple[
    int, int, int, int]:
    """
    Computes tp/fp/tn/fn from teh provided segmentations
    """
    assert prediction.shape == reference.shape, "'prediction' and 'reference' must have the same shape"

    tp = int((roi_mask * (prediction != 0) * (reference != 0)).sum())  # overlap
    fp = int((roi_mask * (prediction != 0) * (reference == 0)).sum())
    tn = int((roi_mask * (prediction == 0) * (reference == 0)).sum())  # no segmentation
    fn = int((roi_mask * (prediction == 0) * (reference != 0)).sum())

    return tp, fp, tn, fn


def dice(tp: int, fp: int, fn: int) -> float:
    """
    Dice coefficient computed using the definition of true positive (TP), false positive (FP), and false negative (FN)
    2TP / (2TP + FP + FN)
    """
    denominator = 2 * tp + fp + fn
    if denominator <= 0:
        return 0

    return 2 * tp / denominator


def jc(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    if denominator <= 0:
        return 0

    return tp / denominator


def hausdorff(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.hd95(prediction, reference)

    except Exception as e:
        print("Hausdorff Error: ", e)
        print(f"Hausdorff prediction does not contain the same label as gt. "
              f"Hausdorff pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return 100


def asd(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.asd(prediction, reference)

    except Exception as e:
        print("ASD Error: ", e)
        print(f"ASD prediction does not contain the same label as gt. "
              f"ASD pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return 100


# Sensitivity: recall
def recall(tp, fn) -> float:
    """TP / (TP + FN) FN: 肿瘤被预测为背景，欠分割"""
    actual_positives = tp + fn
    if actual_positives <= 0:
        return 0
    return tp / actual_positives


def precision(tp, fp) -> float:
    """TP/ (TP + FP)  FP: 背景被预测为肿瘤，过分割"""
    predicted_positives = tp + fp
    if predicted_positives <= 0:
        return 0
    return tp / predicted_positives


def fscore(tp, fp, tn, fn, beta: int = 1) -> float:
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""
    assert beta > 0

    precision_ = precision(tn, fp)
    recall_ = recall(tp, fn)

    if ((beta * beta * precision_) + recall_) <= 0:
        return 0

    fscore = (1 + beta * beta) * precision_ * recall_ / ((beta * beta * precision_) + recall_)
    return fscore

