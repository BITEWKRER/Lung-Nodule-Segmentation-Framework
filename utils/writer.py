# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from functools import singledispatch

import numpy as np
import pandas as pd

from configs import config
from utils.helper import avgStd
from utils.logger import logs


class Writer(object):
    '''
    评估单个结节
    todo 用于保存五折交叉验证的各项精度指标
    '''
    mode = config.mode
    evaluatetype = config.log_name
    csv_path = config.csv_path

    def __init__(self, dataset):
        self.dict = {}
        self.dataset = dataset
        # metric 1
        self.dsc = []
        self.precision = []
        self.sensitivity = []
        self.mIou = []
        self.oneFolddsc = []
        self.oneFoldprecision = []
        self.oneFoldsensitivity = []
        self.oneFoldmIou = []
        # metric 2
        self.HD = []
        self.MSD = []
        self.oneFoldHD = []
        self.oneFoldMSD = []

    @singledispatch
    def __call__(self, dice=None, hd=None, msd=None, avg=False):
        if avg:
            self.oneFoldHD.append(avgStd(self.oneFoldHD))
            self.oneFoldMSD.append(avgStd(self.oneFoldMSD))
            self.oneFolddsc.append(avgStd(self.oneFolddsc))

            for i in range(len(self.oneFolddsc)):
                self.dsc.append(self.oneFolddsc[i])
                self.HD.append(self.oneFoldHD[i])
                self.MSD.append(self.oneFoldMSD[i])

            self.clear()
        else:
            if dice is None and hd is None and msd is None:
                if len(self.oneFolddsc) != 0:
                    # 如果这一fold里面没有对应的标签，那么就求其他fold的均值
                    self.oneFolddsc.append(np.mean(self.oneFolddsc))
                    self.oneFoldMSD.append(np.mean(self.oneFoldMSD))
                    self.oneFoldHD.append(np.mean(self.oneFoldHD))
            else:
                self.oneFolddsc.append(dice)
                self.oneFoldMSD.append(msd)
                self.oneFoldHD.append(hd)

    @singledispatch
    def __call__(self, precision=None, sensitivity=None, dsc=None, mIou=None, avg=False, ):
        if avg:
            # todo 求 现有的五折平均值
            logs(
                f'avg ,Precision : {avgStd(self.oneFoldprecision, log=True)}, '
                f'Sensitivity: {avgStd(self.oneFoldsensitivity, log=True)},'
                f'DSC: {avgStd(self.oneFolddsc, log=True)},'
                f'mIou: {avgStd(self.oneFoldmIou, log=True)}\n')

            self.oneFolddsc.append(avgStd(self.oneFolddsc))
            self.oneFoldmIou.append(avgStd(self.oneFoldmIou))
            self.oneFoldsensitivity.append(avgStd(self.oneFoldsensitivity))
            self.oneFoldprecision.append(avgStd(self.oneFoldprecision))
            # 单个loss的全部精度
            for i in range(len(self.oneFolddsc)):
                self.dsc.append(self.oneFolddsc[i])
                self.precision.append(self.oneFoldprecision[i])
                self.sensitivity.append(self.oneFoldsensitivity[i])
                self.mIou.append(self.oneFoldmIou[i])

            self.clear()
        else:
            if precision is None and sensitivity is None and dsc is None and mIou is None:
                if len(self.oneFoldprecision) != 0:
                    # 如果这一fold里面没有对应的标签，那么就求其他fold的均值
                    self.oneFolddsc.append(np.mean(self.oneFolddsc))
                    self.oneFoldmIou.append(np.mean(self.oneFoldmIou))
                    self.oneFoldsensitivity.append(np.mean(self.oneFoldsensitivity))
                    self.oneFoldprecision.append(np.mean(self.oneFoldprecision))
            else:
                self.oneFolddsc.append(dsc)
                self.oneFoldmIou.append(mIou)
                self.oneFoldsensitivity.append(sensitivity)
                self.oneFoldprecision.append(precision)

    def clear(self, allClear=False):
        if allClear:
            # 置空
            self.oneFolddsc = []
            self.oneFoldprecision = []
            self.oneFoldsensitivity = []
            self.oneFoldmIou = []
            self.dsc = []
            self.precision = []
            self.sensitivity = []
            self.mIou = []

            # metric 2
            self.HD = []
            self.MSD = []
            self.oneFoldHD = []
            self.oneFoldMSD = []

        else:
            # 置空
            self.oneFolddsc = []
            self.oneFoldprecision = []
            self.oneFoldsensitivity = []
            self.oneFoldmIou = []
            self.oneFoldHD = []
            self.oneFoldMSD = []

    @classmethod
    def reshape(cls, arrs):
        print(arrs)
        arr = []
        for i in range(len(arrs)):
            for t in range(len(arrs[i])):
                arr.append(arrs[i][t])

        return arr

    def update(self, model_name):
        data = self.reshape([self.precision, self.sensitivity, self.dsc, self.mIou])

        logs(f'model name {model_name}, len {len(data)} and data len == 6 {len(data) == 6}')
        print(data)
        self.dict.update({f"{model_name}": data})  # todo 一个模型的三个loss的全部精度
        # 置空
        self.clear(True)

    def save(self, model_name):
        df = pd.DataFrame(self.dict)
        '''
            保存 其中一个模型的三种不同loss的各项精度指标
            依次是 精度，敏感度，dice，miou的五折分数及平均值
        '''
        os.makedirs(self.csv_path + '/evaluate/', exist_ok=True)
        df.to_csv(f'{self.csv_path}/evaluate/{model_name}_{self.dataset}_{self.mode}_{self.evaluatetype}_evaluate.csv')
        # 置空
        self.clear(True)
