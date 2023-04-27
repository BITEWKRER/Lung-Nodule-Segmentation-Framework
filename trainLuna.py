# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from configs import config
from utils.trainBase import trainBase


class trainLuna(trainBase):

    def __init__(self, model2d, model3d, lossList):
        super(trainLuna, self).__init__(model2d, model3d, lossList)
        if self.mode == '2d':
            self.seg_path = self.seg_path_luna_2d
        else:
            self.seg_path = self.seg_path_luna_3d
        self.run()


if __name__ == '__main__':

    loss_lists = ['dice', 'bce', 'focal']  #
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet', 'vtunet',
               'pcamnet', 'asa']

    trainLuna(model2d, model3d, loss_lists).to(config.device)
