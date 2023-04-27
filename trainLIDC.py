# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

from configs import config
from utils.EarlyStop import EarlyStopping
from utils.helper import set_init, get_model3d, get_model2d
from utils.loss import Loss
from utils.noduleSet import noduleSet
from utils.trainBase import trainBase


class trainLIDC(trainBase):

    def __init__(self, model2d, model3d, lossList):
        super(trainLIDC, self).__init__(model2d, model3d, lossList)
        self.dataset = 'lidc'
        if self.mode == '2d':
            self.seg_path_Lidc = config.seg_path_lidc_2d
            self.seg_path_Luna = config.seg_path_luna_2d
        else:
            self.seg_path_Lidc = config.seg_path_lidc_3d
            self.seg_path_Luna = config.seg_path_luna_3d

        self.run()

    def initNetwork(self, k):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=1e-4)

        lossf = Loss(self.loss_name)
        scalar = torch.cuda.amp.GradScaler()
        eStop = EarlyStopping(patience=self.earlyEP, fold=int(k), mode=self.mode, path=self.pth_path, verbose=True)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]

       
        lists = set_init(k, self.seg_path_Lidc, None, lists)
        lists = set_init(k, self.seg_path_Luna, None, lists)

        if self.mode == '2d':
            train_dataset = noduleSet(train_list, ['Train', '2d'], self.transform, self.show, )
            train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    num_workers=self.num_worker, pin_memory=True, drop_last=True)
            val_and_test_dataset = noduleSet(val_and_test_list, ['Val', '2d'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=True)
        else:
            train_dataset = noduleSet(train_list, ['Train', '3d'], self.transform, self.show)
            train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    num_workers=self.num_worker, pin_memory=True, drop_last=True)
            val_and_test_dataset = noduleSet(val_and_test_list, ['Val', '3d'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=True)

        return optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter


if __name__ == '__main__':

    loss_lists = ['dice', 'bce', 'focal']
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet', ]

    trainLIDC(model2d, model3d, loss_lists).to('cuda:0')
