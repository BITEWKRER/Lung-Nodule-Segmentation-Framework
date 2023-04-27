# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from configs import GC
from utils.Metrics import Metrics
from utils.helper import get_model2d, get_model3d, set_init, load_model_k_checkpoint
from utils.logger import logs
from utils.noduleSet import noduleSet
from utils.writer import Writer


class evaluateBase(GC):
    """
    评估基类
    """

    def __init__(self, model_lists):
        super(evaluateBase, self).__init__(train=configs.train, dataset=configs.dataset, log_name=configs.log_name,
                                           mode=configs.mode, server=configs.server)
        self.seg_path = None
        self.pth_path = None
        self.model_lists = model_lists
        self.loss_lists = ['dice', 'bce', 'focal']
        logs(f'mode {self.mode}')

    def kFoldMain(self, k, writer, label=None):
        val_and_test_iter, model = self.initNetwork(k, label)
        if val_and_test_iter is not None:
            fprecision, fsensitivity, ff1, fmIou = self.testfn(k, val_and_test_iter, model)
            writer(fprecision, fsensitivity, ff1, fmIou)
        else:
            writer()

    def testfn(self, fold, loader, model):
        model.eval()
        metrics = Metrics().to(self.device)
        with torch.no_grad():
            for idx, data in tqdm(enumerate(loader)):

                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)
                # print(img.shape, msk.shape)
                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), \
                        Variable(msk.cuda(), requires_grad=False)

                preds = model(img)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                metrics(preds, msk)

        model.eval()
        fprecision, fsensitivity, ff1, fmIou = metrics.evluation(fold)
        return fprecision, fsensitivity, ff1, fmIou

    def initNetwork(self, k, label):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]

        lists = set_init(k, self.seg_path, None, lists)

        if label is not None:  # 单个标签类型的评估
            val_list = []
            for item in val_and_test_list:
                if item.find(f'_{label}_') != -1:
                    val_list.append(item)
            val_and_test_list = val_list

        if len(val_and_test_list) != 0:
            # print(len(train_list), len(val_and_test_list))
            val_and_test_dataset = noduleSet(val_and_test_list, ['infer', 'Val'], None, self.show)

            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=True)

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
            return val_and_test_iter, model
        else:
            return None, None

    def run(self, labels=None):
        self.train = False
        if labels is None:  # 整体性能评估
            for model in self.model_lists:
                writer = Writer(self.dataset)
                self.model_name = model
                for loss in self.loss_lists:
                    self.loss_name = loss
                    logs(f'Model {model}, Loss {loss}')
                    for i in range(1, self.k_fold + 1):
                        self.kFoldMain(i, writer, labels, )

                    writer(avg=True)
                writer.update(self.model_name)  # 将不同loss合并为一列
                writer.save(self.model_name)
        elif labels is not None:
            for model in self.model_lists:
                self.model_name = model
                # 逐一评估单一属性
                for label in labels:  # 不同类别
                    writer = Writer(self.dataset)
                    writer.evaluatetype = label
                    for loss in self.loss_lists:
                        self.loss_name = loss
                        logs(f'Model {model}, Loss {loss}, Label {label}')
                        for i in range(1, self.k_fold + 1):
                            self.kFoldMain(i, writer, label)

                        writer(avg=True, )  # 五折交叉验证求均值
                    writer.update(self.model_name)  # 不同loss
                    writer.save(self.model_name)
