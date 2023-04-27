# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from torch.utils.data import DataLoader

from configs import config
from utils.evaluateBase import evaluateBase
from utils.helper import get_model2d, get_model3d, set_init, load_model_k_checkpoint, getAllAttrs
from utils.noduleSet import noduleSet


class evaluateLIDC(evaluateBase):

    def __init__(self, model_lists, labels):
        super(evaluateLIDC, self).__init__(model_lists)

        self.pth_path = config.pth_lidc_path
        self.dataset = 'lidc'

        if self.mode == '2d':
            self.seg_path_luna = config.seg_path_luna_2d
            self.seg_path_lidc = config.seg_path_lidc_2d
        else:
            self.seg_path_luna = config.seg_path_luna_3d
            self.seg_path_lidc = config.seg_path_lidc_3d

        self.run(labels)

    def initNetwork(self, k, label):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]

        lists = set_init(k, self.seg_path_luna, None, lists)
        lists = set_init(k, self.seg_path_lidc, None, lists)

        if label is not None:  # 单个标签类型的评估
            val_list = []
            for item in val_and_test_list:
                if item.find(f'_{label}_') != -1:
                    val_list.append(item)
            val_and_test_list = val_list
        # print(len(val_and_test_list))
        if len(val_and_test_list) != 0:
            val_and_test_dataset = noduleSet(val_and_test_list, ['infer', 'Val'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=True)

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
            return val_and_test_iter, model
        else:
            return None, None


if __name__ == '__main__':
    loss_lists = ['dice', 'bce', 'focal'] 
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet', ]   

    mode = config.mode
    train = config.train  # false
    evaluateLIDC(model3d, None).to('cuda:0')  # 整体评估

    # for labels in getAllAttrs(True).values():  # todo 分项评估
    #     evaluateLIDC(model3d, labels).to(config.device)
