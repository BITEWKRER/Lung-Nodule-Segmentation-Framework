# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

import numpy as np
import torch
import torchmetrics
from torch import nn

from utils.helper import avgStd
from utils.logger import logs

np.seterr(divide='ignore', invalid='ignore')

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class Metrics(nn.Module):

    def __init__(self):
        super(Metrics, self).__init__()
        self.eps = 1e-5

        self.precision = []
        self.f1_score = []
        self.mIou = []
        self.sensitivity = []

    def __call__(self, preds, labels):
        preds = preds.type(torch.FloatTensor)
        labels = labels.type(torch.IntTensor)
        # todo 非库方法
        # precision, recall, dice, pixel_acc, specificity, mIou = self._calculate_overlap_metrics(labels, preds)

        self.sensitivity.append(self.numDeal(torchmetrics.functional.recall(preds, labels).data, self.sensitivity))
        self.precision.append(self.numDeal(torchmetrics.functional.precision(preds, labels).data, self.precision))
        self.mIou.append(self.numDeal(torchmetrics.functional.jaccard_index(preds, labels, 2).data, self.mIou))
        self.f1_score.append(self.numDeal(torchmetrics.functional.f1_score(preds, labels).data, self.f1_score))

    def numDeal(self, nums, lists):
        if np.isnan(nums):
            logs('exist nan')
            nums = np.mean(lists)
        return np.round((nums * 100), 2)  # 百分制，小数点后两位

    def evluation(self, fold):
        logs(
            f"Fold {fold}"
            f",Precision : " + avgStd(self.precision, log=True) +
            f",Sensitivity: " + avgStd(self.sensitivity, log=True) +
            f",dsc: " + avgStd(self.f1_score, log=True) +
            f",mIou: " + avgStd(self.mIou, log=True)
        )
        return avgStd(self.precision), avgStd(self.sensitivity), avgStd(self.f1_score), avgStd(self.mIou)
