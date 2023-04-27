# -*-coding:utf-8 -*-
"""
# Time       ：2022/4/17 14:44
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import pandas as pd
from torch import nn

from configs import config
from utils.helper import getAllAttrs


class reader(nn.Module):
    """
    一列模型评估结果坐标构成
            dice    bce     focal
    pre     5       11      17
    sen     23      29      35
    dsc     41      47      53
    mIou    59      65      71
    """
    csv_path = config.csv_path
    mode = config.mode

    def __init__(self, dataset, losses=None):
        super(reader, self).__init__()
        self.dict = dict()
        self.dataset = dataset
        self.losses = losses
        if losses is None:
            self.losses = ['dice', ]  # 'bce', 'focal'

    def loadOneAttrs(self, labels, model_name):
        """
        加载一个属性中全部的label
        """
        datas = []
        for label in labels:
            path = f'{self.csv_path}/evaluate/{model_name}_{self.dataset}_{self.mode}_{label}_evaluate.csv'
            print(path)
            data = pd.read_csv(path, header=0)
            datas.append(data)
        return datas

    def loadEntirety(self, size=6):
        """
        加载全部模型的四个平均值表精度
        """
        data = pd.read_csv(f'{self.csv_path}/evaluate/{self.dataset}_{self.mode}_eva_all_evaluate.csv', header=0)
        preBase = 5
        senBase = 23
        dscBase = 41
        mIouBase = 59

        for model in data.columns[1:]:
            for i, loss in enumerate(self.losses):
                print(f'loss {loss}:', [data[model][preBase + i * size], data[model][senBase + i * size],
                                        data[model][dscBase + i * size], data[model][mIouBase + i * size]])

    def combination(self, key, labels, model_name, size=6, baseIdx=41):
        """
        将同一label的不同属性的dsc属性进行组合，其中包括不同模型的不同loss
        """
        print(key)
        datas = self.loadOneAttrs(labels, model_name)
        for model in datas[0].columns[1:]:
            for i, loss in enumerate(self.losses):
                tmplist = []
                for data in datas:  # 不同csv
                    tmplist.append(data[model][(i * size) + baseIdx])
                    """
                    输出长度与label
                    """
                self.dict.update({f'{model}_{loss}': tmplist})
        print(self.dict)


if __name__ == '__main__':
    # reader('luna', ).loadEntirety()
    model_name = 'raunet'
    labels = getAllAttrs(True)
    for key, label in labels.items():
        reader('lidc', ).combination(key, label, model_name)
