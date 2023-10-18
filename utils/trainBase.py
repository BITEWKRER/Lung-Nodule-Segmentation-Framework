# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
import time

import numpy as np
import torch.nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from configs import GC
from utils.EarlyStop import EarlyStopping
from utils.Metrics import Metrics
# from utils.MetricsV2 import MetricsV2
from utils.helper import transforms2d, transforms3d, get_model2d, get_model3d, set_init, avgStd, showTime, save_tmp, \
    load_model_k_checkpoint
from utils.logger import logs
from utils.loss import Loss
from utils.noduleSet import noduleSet


def dice_sum(preds, msk, loss_fn):
    loss0 = loss_fn(preds[0], msk)
    loss1 = loss_fn(preds[1], msk)
    loss2 = loss_fn(preds[2], msk)
    loss3 = loss_fn(preds[3], msk)
    loss4 = loss_fn(preds[4], msk)
    loss = loss0 + loss1 + loss2 + loss3 + loss4

    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f\n" % (
    #     # loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))
    #     loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()))
    return loss0, loss


class trainBase(GC):
    """
    训练基类
    """
    seg_path = None
    model_name = None

    def __init__(self, model2d, model3d, lossList):
        super(trainBase, self).__init__(train=configs.train, dataset=configs.dataset, log_name=configs.log_name,
                                        mode=configs.mode, pathV=configs.pathV, LossV=configs.LossV,
                                        FileV=configs.FileV, MetricsV=configs.MetricsV, sup=configs.sup,
                                        server=configs.server)
        self.model2d = model2d
        self.model3d = model3d
        self.lossList = lossList

        if self.mode == '2d':
            self.transform = transforms2d
            self.models = self.model2d
        else:
            self.transform = transforms3d
            self.models = self.model3d

        if self.dataset == 'luna':
            self.pth_path = self.pth_luna_path
        else:
            self.pth_path = self.pth_lidc_path

    def baseInfo(self):
        logs(
            f" Dataset:{self.dataset},\n"
            f' Train batch:{str(self.train_batch_size)},\n'
            f' Val and Test batch:{str(self.val_and_test_batch_size)},\n'
            f' Optimizer:{self.optimizer},\n'
            f' lr:{self.lr},\n'
            f' k_fold:{self.k_fold},\n'
            f' num worker:{self.num_worker},\n'
            f' early stop:{self.earlyEP},\n'
            f' mode:{self.mode},\n'
            f' device:{self.device},\n'
            f' epoches:{self.epochs},\n'
            f' seg path:{self.seg_path},\n'
            f' pth path:{self.pth_path},\n'
            f' sup :{self.sup}'
        )

    def initNetwork(self, k):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=1e-4)

        scalar = torch.cuda.amp.GradScaler()
        eStop = EarlyStopping(patience=self.earlyEP, fold=int(k), mode=self.mode, path=self.pth_path, verbose=True)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]

        lossf = Loss(self.loss_name)

        lists = set_init(k, self.seg_path, None, lists)

        if self.mode == '2d':
            train_dataset = noduleSet(train_list, ['Train', '2d'], self.transform, self.show, )
            train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    num_workers=self.num_worker, pin_memory=True, drop_last=True)
            val_and_test_dataset = noduleSet(val_and_test_list, ['Val', '2d'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=False, drop_last=True)
        else:
            train_dataset = noduleSet(train_list, ['Train', '3d'], self.transform, self.show)
            train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    num_workers=self.num_worker, pin_memory=True, drop_last=True)
            val_and_test_dataset = noduleSet(val_and_test_list, ['Val', '3d'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=False, drop_last=True)

        return optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter

    def kFoldTrain(self, k, eltSet):
        optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter = self.initNetwork(k)

        if configs.train:
            '''训练'''
            for ep in range(1, self.epochs + 1):
                if not eStop.early_stop and not self.lost_loss:
                    train_fps = self.trainFun(ep, train_iter, model, optimizer, lossf, scalar)
                    eltSet[4].append(float(train_fps))
                    self.validationFun(val_and_test_iter, model, lossf, ep, eStop)
                else:
                    eltSet[6].append(float(eStop.epoch))
                    break

        """梯度没有消失或者爆炸,进行评估"""
        if not self.lost_loss:
            # todo =========================一折模型评估，推理时间=======================
            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name,
                                    model, k)

            k_fold_eva = self.testFun(k, val_and_test_iter, model, mode='evluation')
            for t in range(len(k_fold_eva)):
                eltSet[t].append(float(k_fold_eva[t]))

            # todo =========================一折模型推理时间============================
            infer_fps = self.testFun(k, val_and_test_iter, model)
            eltSet[5].append(float(infer_fps))
            logs(f'Fold {k} Infer {infer_fps:.2f} FPS')
        self.lost_loss = False
        return eltSet

    def trainFun(self, ep, train_iter, model, optimizer, loss_fn, scalar):
        times = []
        loop = tqdm(train_iter)
        for idx, data in enumerate(loop):
            start_time = time.time()

            img, msk = data['img'], data['msk']
            img = img.type(torch.FloatTensor)
            msk = msk.type(torch.FloatTensor)

            if self.device != 'cpu' and torch.cuda.is_available():
                img, msk = Variable(img.cuda(), requires_grad=False), \
                    Variable(msk.cuda(), requires_grad=False)

            with torch.cuda.amp.autocast():

                preds = model(img)
                loss = loss_fn(preds, msk)

                optimizer.zero_grad()
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()

                if np.isnan(loss.cpu().item()):
                    self.lost_loss = True
                    break

                loop.set_postfix(loss=loss.item())

                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

        fps = 1.0 / np.mean(times)
        return fps

    def validationFun(self, loader, model, loss_fn, ep, eStop):
        model.eval()
        metrics = []
        os.makedirs(self.pred_path, exist_ok=True)
        with torch.no_grad():
            for idx, data in tqdm(enumerate(loader)):
                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)

                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), \
                        Variable(msk.cuda(), requires_grad=False)

                preds = model(img)
                loss = loss_fn(preds, msk)

        metrics = np.asarray(metrics, np.float32)
        eStop(np.mean(metrics), model, ep, self.model_name, self.optimizer, self.loss_name)
        model.train()

    def testFun(self, k_fold, loader, model, mode='infer'):
        model.eval()
        metrics = Metrics().to(self.device)
        times = []

        with torch.no_grad():
            for idx, data in tqdm(enumerate(loader)):
                test_start_time = time.time()

                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)

                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), \
                        Variable(msk.cuda(), requires_grad=False)

                if self.sup:
                    preds = model(img)
                    preds = preds[0]
                else:
                    preds = model(img)

                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()

                test_end_time = time.time()
                times.append(test_end_time - test_start_time)

                if mode == 'evluation':
                    metrics(preds, msk)
                if idx % 50 == 0 and self.mode == '2d':
                    save_tmp(self.pred_path, img[0], msk[0], preds[0], 'test_tmp')

        model.train()
        if mode == 'evluation':
            fprecision, fsensitivity, ff1, fmIou = metrics.evluation(k_fold)
            return [fprecision, fsensitivity, ff1, fmIou]
        else:
            fps = 1.0 / np.mean(times)
            return fps

    def run(self):
        self.baseInfo()
        for model in self.models:
            self.model_name = model
            for loss in self.lossList:
                self.loss_name = loss
                start_time = time.time()

                precision, sensitivity, f1, mIou, train_times, infer_times, oc = [], [], [], [], [], [], []
                eltSet = [precision, sensitivity, f1, mIou, train_times, infer_times, oc]

                for i in range(1, self.k_fold + 1):
                    logs(f'Fold {i}')
                    eltSet = self.kFoldTrain(i, eltSet)

                logs(
                    f'Final '
                    f'Precision:{avgStd(precision, log=True)},'
                    f'Sensitivity:{avgStd(sensitivity, log=True)},'
                    f'dsc:{avgStd(f1, log=True)},'
                    f'mIou:{avgStd(mIou, log=True)},'
                    f'{self.dataset} fps:{avgStd(train_times, log=True)},'
                    f'infer fps:{avgStd(infer_times, log=True)},'
                    f'Optimal CVG:{avgStd(oc, log=True)},'
                )
                end_time = time.time()
                showTime('Total', start_time, end_time)
