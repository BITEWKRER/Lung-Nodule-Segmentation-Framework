# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os

import numpy as np
import torch
from torch import nn

from utils.Metrics import logs


class EarlyStopping(nn.Module):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, fold, patience=7, mode='2d', verbose=False, delta=0.0, path='./'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.fold = fold
        self.epoch = 0.
        self.mode = mode

        os.makedirs(self.path, exist_ok=True)

    def __call__(self, val_loss, model, epoch, model_name, optimizer, loss_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name, optimizer, loss_name)
            self.epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name, optimizer, loss_name)
            self.counter = 0
            self.epoch = epoch
            # 调试
            # self.early_stop = True

    def save_checkpoint(self, val_loss, model, model_name, optimizer, loss_name):
        '''Saves model when validation loss decrease.'''

        if self.verbose:
            logs(
                f'Best Epoch {int(self.epoch)}, Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  '
                f'Saving model ...')
        torch.save(model.state_dict(),
                   self.path + f'/{self.mode}_{model_name}_{str(self.fold)}_{optimizer}_{loss_name}_checkpoint.pth')

        self.val_loss_min = val_loss
