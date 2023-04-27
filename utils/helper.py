# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

import os
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.utils
from batchgenerators.augmentations.utils import resize_segmentation
from matplotlib import pyplot as plt
from ptflops import get_model_complexity_info
from skimage.transform import resize

from configs import config
from models.model_trans.transbts.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models.model_trans.uctrans.UCTRANSNET import UCTransNet
from models.model_trans.uctrans.uctransnetconfig import get_CTranS_config
from models.model_trans.unetr import UNETR
from models.models_2d.mipt.cpfnet import CPFNet
from models.models_2d.mipt.raunet import RAUNet
from models.models_2d.mipt.sgunet import SGU_Net
from models.models_2d.mipt.unet3p import UNet_3Plus
from models.models_2d.mipt.unext import UNext
from models.models_2d.others.unet2 import U_Net
from models.models_3d.mipt.reconnet import ReconNet
from models.models_3d.mipt.vnet import VNet
from models.models_3d.mipt.wingsnet import WingsNet
from models.models_3d.mipt.ynet3d import YNet3D
from models.models_3d.others.resunet.model import ResidualUNet3D, UNet3d
from utils.logger import logs

pth_path = config.pth_luna_path
pred_path = config.pred_path


def getAllAttrs(evaluate=False):
    attrs = dict()
    subtlety = ['ExtremelySubtle', 'ModeratelySubtle', 'FairlySubtle', 'ModeratelyObvious', 'Obvious']
    internalStructure = ['SoftTissue', 'Fluid', 'Fat', 'Air']
    calcification = ['Popcorn', 'Laminated', 'cSolid', 'Noncentral', 'Central', 'Absent']
    sphericity = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']
    margin = ['PoorlyDefined', 'NearPoorlyDefined', 'MediumMargin', 'NearSharp', 'Sharp']
    lobulation = ['NoLobulation', 'NearlyNoLobulation', 'MediumLobulation', 'NearMarkedLobulation',
                  'MarkedLobulation']
    spiculation = ['NoSpiculation', 'NearlyNoSpiculation', 'MediumSpiculation', 'NearMarkedSpiculation',
                   'MarkedSpiculation']
    texture = ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed', 'SolidMixed', 'tSolid']
    maligancy = ['benign', 'uncertain', 'malignant']

    if evaluate:
        attrs.update({'subtlety': subtlety})

        # attrs.update({'internalStructure': internalStructure}) # 不评估该属性  x
        # attrs.update({'calcification': calcification})  x

        sphericity = ['OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']  # luna Linear 不存在
        # sphericity = ['Linear']
        attrs.update({'sphericity': sphericity})
        attrs.update({'margin': margin})
        attrs.update({'lobulation': lobulation})
        attrs.update({'spiculation': spiculation})
        # texture = ['subsolid', 'tSolid']  # 手动替换 x
        attrs.update({'texture': texture})
        attrs.update({'maligancy': maligancy})
        size = ['sub36', 'sub6p', 'solid36', 'solid68', 'solid8p']  # 投票时手动指定
        attrs.update({'size': size})
    else:
        attrs = [subtlety, internalStructure, calcification, sphericity, margin, lobulation,
                 spiculation, texture]

    return attrs


def get_set(k, lesion_list):
    set_len = len(lesion_list)
    copies = int(set_len * config.val_domain)

    val_sidx = (k - 1) * copies
    val_eidx = val_sidx + copies
    if k == 5:
        val_eidx = max(val_eidx, set_len)
    # todo 得到验证集
    val_set = lesion_list[val_sidx:val_eidx]

    train_set = []
    train_set.extend(lesion_list[:val_sidx])
    train_set.extend(lesion_list[val_eidx:])

    return [train_set, val_set]


def set_init(k, seg_path, re, lists, format='*.npy'):
    if re is not None:
        lesion_list = glob(seg_path + re)
        lesion_list.sort()
    else:
        # lesion_list = glob(seg_path + '*.nii.gz')
        print(len(glob(seg_path + format)))
        lesion_list = glob(seg_path + format)
        lesion_list.sort()
    lesion_list = [item for item in lesion_list if 'sub3c' not in item]
    lesion_list = [item for item in lesion_list if 'solid3c' not in item]

    if len(lesion_list) != 0:
        set_list = get_set(k, lesion_list)
        for i in range(len(set_list)):
            lists[i].extend(set_list[i])
        return lists
    else:
        return lists


def save_tmp(path, _img, _msk, _pred, name):
    _img = _img.cpu().numpy()
    _msk = _msk.cpu().numpy()
    _pred = _pred.cpu().numpy()

    fig, plots = plt.subplots(1, 3)
    plots[0].imshow(_img[0], cmap='gray')
    plots[1].imshow(_msk[0], cmap='gray')
    plots[2].imshow(_pred[0], cmap='gray')
    plots[0].axis('off')
    plots[1].axis('off')
    plots[2].axis('off')
    plt.tight_layout()
    plt.savefig(path + '/' + name + '.png', pad_inches=0)
    plt.close()


def transforms2d(img, label, flip=False):
    if flip:
        rn = random.random()
        if rn < 0.35:  # 水平翻转
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)
        elif rn < 0.75:  # 垂直
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

    return img, label


def transforms3d(img, label, flip=False):
    if flip:
        cnt = 3
        while random.random() < 0.5 and cnt > 0:
            degree = random.choice([0, 1, 2])
            img = np.flip(img, axis=degree)
            label = np.flip(label, axis=degree)
            cnt -= 1

    return img, label


# todo 求均值和方差
def avgStd(arr, log=False):
    arr = np.array(arr)
    if log:
        return f"{np.round(arr.mean(), 2)}±{np.round(arr.std(ddof=1), 2)}"
    else:
        return np.round(arr.mean(), 2)


def showTime(fold, start, end):
    times = round(end - start, 2)
    if fold not in [1, 2, 3, 4, 5]:
        logs(f'Fold {fold},time:'
             f"{times:.3f}s,"
             f"{times / 60 :.3f} mins")
    else:
        logs(f'{fold},time:'
             f"{times:.3f}s,"
             f"{times / 60 :.3f} mins")
    return times


# # 重采样：原始CT分辨率往往不一致，为便于应用网络，需要统一分辨率
def resample2d(imgs, spacing, new_spacing=[1., 1.], is_seg=False):
    new_shape = np.round(((np.array(spacing) / np.array(new_spacing)).astype(float) * imgs.shape)).astype(int)
    if is_seg:
        order = 0
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
        imgs = resize_fn(imgs, new_shape, order)
    else:
        order = 3
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
        imgs = resize_fn(imgs, new_shape, order, **kwargs)

    return imgs


def fliter(imgs, masks, verbose=False):
    # 过滤 黑色背景
    ti = np.zeros_like(imgs)
    tm = np.zeros_like(masks)
    t = 0
    for i in range(imgs.shape[2]):
        a = imgs[:, :, i]
        if np.count_nonzero(a) != 0:
            ti[:, :, t] = imgs[:, :, i]
            tm[:, :, t] = masks[:, :, i]
            t += 1
    imgs = ti[:, :, :t]
    masks = tm[:, :, :t]
    if verbose:
        print(f'now {imgs.shape, masks.shape}')
    return imgs, masks


def img_crop_or_fill(img, mode='2d'):
    """
    避免出现填充的情况，即结节窗口取得足够大，有足够的窗口完成裁剪
    """
    if mode == '2d':
        size = config.input_size_2d
        if img.shape[0] > size and img.shape[1] > size:
            img = center_crop(img, size, size)
        else:
            img = fill(img, size)
    elif mode == '3d':
        size = config.input_size_3d
        if img.shape[0] > size and img.shape[1] > size and img.shape[2] > size:
            img = center_crop(img, size, size, size)
        else:
            img = fill(img, size)

    return img


def gapCal(img_size, size):
    gap = 0
    if img_size > size:
        gap = int(np.ceil((img_size - size) / 2))
    return gap


def fill(img, size):
    if len(img.shape) == 2:
        arr = np.zeros([size, size])
        gapx = gapCal(img.shape[0], size)
        gapy = gapCal(img.shape[1], size)
        img = img[gapx:img.shape[0] - gapx, gapy:img.shape[1] - gapy]
        arr[:img.shape[0], :img.shape[1]] = img
    elif len(img.shape) == 3:
        arr = np.zeros([size, size, size])
        gapx = gapCal(img.shape[0], size)
        gapy = gapCal(img.shape[1], size)
        gapz = gapCal(img.shape[2], size)
        img = img[gapx:img.shape[0] - gapx, gapy:img.shape[1] - gapy, gapz:img.shape[2] - gapz]
        arr[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    else:
        raise RuntimeError('wrong shape')
    return arr


def center_crop(img, new_width=None, new_height=None, new_z=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        z = img.shape[2]
        up_z = int(np.ceil((z - new_z) / 2))
        floor_z = z - int(np.floor((z - new_z) / 2))
        center_cropped_img = img[top:bottom, left:right, up_z:floor_z]

    return center_cropped_img


def load_model_k_checkpoint(pthPath, mode, model_name, optimizer, loss_name, model, k, verbose=True):
    if verbose:
        logs(f'============load {model_name} == Fold {k} check point============')
    file = pthPath + f'/{mode}_{model_name}_{str(k)}_{optimizer}_{loss_name}_checkpoint.pth'
    print(file)
    if not os.path.exists(file):
        logs('pth not exist')
        exit(0)
    else:
        if torch.cuda.is_available():
            check_point = torch.load(file)
        else:
            check_point = torch.load(file, map_location=torch.device('cpu'))
        """
        问题描述：
        变量中含有  环境 不一致导致 （cuda 和 CPU）
        """
        try:
            model.load_state_dict(check_point)
        except:
            print('strict false')
            model.load_state_dict(check_point, False)


def save_predictions_as_imgs(loader, model, folder=pred_path, device='cuda:0', verbose=True):
    if verbose:
        logs('============save prediction============')
    os.makedirs(pred_path, exist_ok=True)
    model.eval()
    for idx, data in enumerate(loader):
        name, img, msk = data['name'], data['image'], data['mask']

        img = img.type(torch.FloatTensor)
        msk = msk.type(torch.FloatTensor)
        img = img.to(device)
        msk = msk.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))  # 判断是否进行了softmax 或者sigmoid
            preds = (preds > 0.5).float()
        logs(preds.shape)

        res = preds
        torchvision.utils.save_image(res, f"{folder}/predict_{name[0]}.png")
    model.train()


def get_parm(model='2d', model_name='None', verbose=False):
    model = model.lower()
    print(model_name)
    if model == '2d':
        SIZE = config.input_size_2d

        model = get_model2d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)
    else:
        SIZE = config.input_size_3d

        model = get_model3d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE, SIZE,), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params


def get_model2d(model_name, device, verbose=False):
    if model_name == 'unet':
        model = U_Net(1, 1)
    elif model_name == 'unetpp':
        model = smp.UnetPlusPlus(classes=1, in_channels=1)
    elif model_name == 'unet3p':
        model = UNet_3Plus()
    elif model_name == 'cpfnet':
        model = CPFNet()
    elif model_name == 'raunet':
        model = RAUNet()
    elif model_name == 'sgunet':
        model = SGU_Net(1, 1)
    elif model_name == 'uctransnet':
        model = UCTransNet(get_CTranS_config())
    elif model_name == 'unext':
        model = UNext(1, input_channels=1, in_chans=1, img_size=64)
    else:
        raise Exception(f"no model name as {model_name}")

    if config.device != 'cpu' and torch.cuda.is_available():
        if verbose:
            logs(f'Use {device}')
        model.to(device)
    else:
        if verbose:
            logs('Use CPU')
        model.to('cpu')
    return model


def get_model3d(model_name, device, verbose=False):
    model_name = model_name.lower()

    if model_name == 'unet':
        model = UNet3d(1, 1, False, )
    elif model_name == 'resunet':
        model = ResidualUNet3D(1, 1, False, )
    elif model_name == 'vnet':
        model = VNet()
    elif model_name == 'ynet':
        model = YNet3D()
    elif model_name == 'wingsnet':
        model = WingsNet()
    elif model_name == 'reconnet':
        model = ReconNet(32, 1)
    elif model_name == 'unetr':
        model = UNETR(in_channels=1, out_channels=1, img_size=(64, 64, 64), pos_embed='conv', norm_name='instance')
    elif model_name == 'transbts':
        _, model = TransBTS(img_dim=64, num_classes=1)  # todo 修改位置编码 4096-》512
    else:
        raise Exception(f"no model name as {model_name}")

    if config.device != 'cpu' and torch.cuda.is_available():
        if verbose:
            logs(f'Use {device}')
        model.to(device)
    else:
        if verbose:
            logs('Use CPU')
        model.to('cpu')
    return model


if __name__ == '__main__':
    # TODO
    model2d = ['unet', 'unetpp', 'raunet', 'cpfnet', 'unet3p', 'sgunet', 'sgl',
               'bionet', 'kiunet', 'msnet']

    model3d = ['unet3d', 'vnet', 'ynet', 'kiunet3d', 'unetpp3d', 'wingsnet', ]

    # todo 需要重新设置
    modelTrans = ['uctransnet', 'medt', 'utnet', 'swinunet', 'transbts', 'unetr', ]

    get_parm('3d', 'unetr', True)
