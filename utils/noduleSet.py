# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import config
from utils.logger import logs


class noduleSet(Dataset):
    def __init__(self, lists, mode, transform=None, show=False):
        super(noduleSet, self).__init__()
        self.show = show
        self.transform = transform
        self.lists = lists
        self.mode = mode
        logs(f'{mode[0]},{mode[1]} len: {len(self.lists)}')

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):

        img_name = self.lists[idx].split('.')[-2]

        data = np.load(self.lists[idx])

        if self.transform is not None:  # 图像增强
            data[0], data[1] = self.transform(data[0], data[1], True)

        data = torch.as_tensor(data.copy()).float().contiguous()
        img, msk = data.split(1, dim=0)

        if self.show:
            fig, plots = plt.subplots(1, 2)
            if self.mode[1] == '2d':
                plots[0].imshow(img[0], cmap='gray')
                plots[1].imshow(msk[0], cmap='gray')
            else:
                plots[0].imshow(img[:, :, 0], cmap='gray')
                plots[1].imshow(msk[:, :, 0], cmap='gray')
            plt.show()

        return {
            'name': img_name,
            'img': torch.as_tensor(img).float().contiguous(),
            'msk': torch.as_tensor(msk).float().contiguous()
        }
