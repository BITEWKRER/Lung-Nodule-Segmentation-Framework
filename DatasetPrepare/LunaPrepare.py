# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from __future__ import absolute_import

import multiprocessing
import os
import random
import sys

# todo 无法运行cmd时，取消注释下一行
from utils.logger import logs

sys.path.append(os.pardir)  # 环境变量
import time
from functools import partial
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from torch import nn

from configs import config
from utils.helper import showTime, fliter, img_crop_or_fill, resample2d, getAllAttrs
from utils.resample.resample import resample_patient


class FindLunaNodule(nn.Module):
    """
    通过结节点xyz与lidc的结节进行对比,依次找到对应结节
    """
    seg_path3d = None
    seg_path2d = None
    dataset = 'luna'

    def dirInit(self):
        self.seg_path3d = config.seg_path_luna_3d
        self.seg_path2d = config.seg_path_luna_2d

        os.makedirs(self.seg_path2d, exist_ok=True)
        os.makedirs(self.seg_path3d, exist_ok=True)
        print(f'dir init in {self.seg_path2d},{self.seg_path3d}')

    @classmethod
    def findAllRow(cls, data):
        # todo 根据对应csv文件找出对应的idx
        data = data.groupby('seriesuid').apply(
            lambda d: tuple(d.index) if len(d.index) > 0 else None
        ).dropna()
        return data

    @classmethod
    def load_itk_image_simple(cls, filename):

        itkimage = sitk.ReadImage(filename)

        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
        return numpyOrigin, numpySpacing

    @classmethod
    def worldToVoxelCoord(cls, worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    def findmax(self, imgs, masks):
        # todo 找出2d mask中结节最大的一个
        imgs = np.array(imgs)
        masks = np.array(masks)

        cnt = []
        for t in range(imgs.shape[2]):
            n = np.count_nonzero(masks[:, :, t])
            cnt.append(n)

        idx = self.maxidx(cnt)
        return imgs[:, :, idx], masks[:, :, idx], idx

    @classmethod
    def maxidx(cls, a):
        a = np.array(a)
        return np.where(a == np.max(a))[0][0]

    @classmethod
    def lumTrans(cls, img):
        """
        截断归一化，只关注兴趣区域
        """
        lungwin = np.array([-1000., 400.])
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        return newimg

    @classmethod
    def vote(cls, values):
        """
        通过投票决定四个医生的标注属性的结节最终属性
        """
        cnt = dict()
        for val in values:
            if str(val) in cnt.keys():
                cnt.update({f'{val}': cnt.get(f'{val}') + 1})
            else:
                cnt.update({f'{val}': 1})

        # todo 检查是否有相同大小的值
        highest = max(cnt.values())
        idxs = [int(k) for k, v in cnt.items() if v == highest]
        if len(idxs) == 1:
            return idxs[0] - 1, idxs[0]
        else:
            return int(np.floor(np.median(idxs))) - 1, int(np.floor(np.median(idxs)))

    def voteAllAttr(self, arrs, features):
        labels = getAllAttrs()
        result = []
        for i, arr in enumerate(arrs):
            labelIdx, doctorIdx = self.vote(arr)
            features[i] = doctorIdx
            result.append(labels[i][labelIdx])
        return result, features

    def mix(self, anns, lunaDiameters, UseLunaDiameter=True):
        """
        良恶性判断使用中位数判断，大于3为恶性，等于3为不确定，小于3为良性
        直径：不同掩码直径均值。luna实际测量值
        结节大小分类：
            实性结节(Solid)：3~6，6~8，>8
            半实性结节(Subsolid Nodules)：3~6，>=6
        其余属性均进行投票
        """
        features_name = ('subtlety',
                         'internalStructure',
                         'calcification',
                         'sphericity',
                         'margin',
                         'lobulation',
                         'spiculation',
                         'texture',
                         'malignancy')
        diameters = []
        subtlety = []
        internalStructure = []
        calcification = []
        sphericity = []
        margin = []
        lobulation = []
        spiculation = []
        texture = []
        malignancy = []
        centroid = np.zeros(3)
        features = np.zeros(9)

        for ann in anns:
            # attr
            subtlety.append(ann.subtlety)
            internalStructure.append(ann.internalStructure)
            calcification.append(ann.calcification)
            sphericity.append(ann.sphericity)
            margin.append(ann.margin)
            lobulation.append(ann.lobulation)
            spiculation.append(ann.spiculation)
            texture.append(ann.texture)
            malignancy.append(ann.malignancy)
            diameters.append(ann.diameter)
            centroid += ann.centroid

        centroid /= len(anns)
        arrs = [subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture]
        result, features = self.voteAllAttr(arrs, features)

        """
        使用luna结节直径进行判断结节大小
        """
        if UseLunaDiameter:
            # 使用luna的直径
            avgDiameter = lunaDiameters
        else:
            # 使用官方预估的直径
            avgDiameter = np.average(diameters)

        noduleSize = 'unassignment'
        if result[7] in ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed']:  # 改变结节名称
            result[7] = 'subsolid'  # 部分实性
            if 3. <= avgDiameter < 6:
                noduleSize = 'sub36'
            elif avgDiameter >= 6:
                noduleSize = 'sub6p'
            elif avgDiameter < 3.:
                noduleSize = 'sub3c'
        elif result[7] in ['SolidMixed', 'tSolid']:
            result[7] = 'tSolid'  # 实性
            if 3. <= avgDiameter < 6.:
                noduleSize = 'solid36'
            elif 6. <= avgDiameter <= 8.:
                noduleSize = 'solid68'
            elif avgDiameter > 8:
                noduleSize = 'solid8p'
            elif avgDiameter < 3.:
                noduleSize = 'solid3c'
        else:
            logs(f'error {result[7]}')

        """
        中位数决定结节良恶性
        """
        malignancy.sort()
        malignancy = np.round(np.median(malignancy), 2)
        features[8] = malignancy
        if malignancy < 3:
            malignancy = 'benign'
        elif malignancy == 3:
            malignancy = 'uncertain'
        else:
            malignancy = 'malignant'
        result.append(malignancy)

        return result, noduleSize, avgDiameter, features, centroid

    def main(self, idx, mode=None):
        data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
        mhd_origin_spacing = pd.read_csv(f'{config.csv_path}/MhdOriginAndSpaving.csv', header=0)

        count = self.findAllRow(data)  # 统计id下的所有结节数量
        orgins_count = self.findAllRow(mhd_origin_spacing)

        attrs = []
        # todo 对相应点的结节进行掩码聚类
        all_nodules = []
        # 多线程并发
        size = 18
        start = size * idx
        end = start + size
        if idx == 33:
            end = None
        if idx == -1:
            start = 0
            end = None

        print(f'Thread : {idx}, start {start} to {end}')
        for i, item in (enumerate(np.unique(data['seriesuid'])[start:end])):
            if i >= 601:
                break
            anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.series_instance_uid == item).all()
            CT = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == item).first()
            if mode is not None:
                vol = CT.to_volume(verbose=False)
                vol = self.lumTrans(vol)

            spacing = CT.spacings
            slice_thickness = CT.slice_thickness
            pixel_spacing = CT.pixel_spacing
            patient_id = CT.patient_id

            ann_ex = []
            idx = orgins_count[item][0]
            mhdOrigin = np.array([mhd_origin_spacing['originX'][idx], mhd_origin_spacing['originY'][idx],
                                  mhd_origin_spacing['originZ'][idx]], dtype=np.float64)
            mhdSpacing = np.array([mhd_origin_spacing['spacingX'][idx], mhd_origin_spacing['spacingY'][idx],
                                   mhd_origin_spacing['spacingZ'][idx]], dtype=np.float64)
            for row in count[item]:
                xyz = np.array([data['coordX'][row], data['coordY'][row], data['coordZ'][row]], dtype=np.float64)
                xyz = np.round(self.worldToVoxelCoord(xyz[::-1], mhdOrigin, mhdSpacing)[::-1], 2)
                x, y, z = xyz[0], xyz[1], xyz[2]

                one_nodule = []
                """遍历serious id中所有掩膜,比较xyz,将在bias内的结节进行合并"""
                for k, ann in enumerate(anns):
                    if k not in ann_ex:
                        v1 = np.round(ann.centroid[0], 2)
                        v2 = np.round(ann.centroid[1], 2)
                        v3 = np.round(ann.centroid[2], 2)
                        diffz = abs(z - v3)
                        diffy = abs(y - v1)
                        diffx = abs(x - v2)
                        bias = 5.  # 中心位置偏差
                        if len(anns) == 3 or item in [  # 特殊结节
                            '1.3.6.1.4.1.14519.5.2.1.6279.6001.321935195060268166151738328001',
                            '1.3.6.1.4.1.14519.5.2.1.6279.6001.286422846896797433168187085942'
                        ]:
                            if diffx <= bias * 2 and diffy <= bias * 2 and diffz < 8.8:  # 一张切片只有三个标注，可能都是
                                ann_ex.append(k)
                                one_nodule.append(ann)
                        else:
                            """包含三层切片，如果还取不到，则认为没有同一个结节"""
                            if diffx <= bias and diffy <= bias and diffz < bias:
                                ann_ex.append(k)
                                one_nodule.append(ann)

                # todo 保存不同掩码均值属性,获取该结节的相关属性信息
                result, noduleSize, avgDiameter, features, centroid = self.mix(one_nodule, data['diameter_mm'][row])

                attrs.append(
                    {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1], 'centroidZ': centroid[2],
                     'diameter': avgDiameter, 'subtlety': features[0], 'internalStructure': features[1],
                     'calcification': features[2], 'sphericity': features[3], 'margin': features[4],
                     'lobulation': features[5], 'spiculation': features[6], 'texture': features[7],
                     'malignancy': features[8], 'spacingX': spacing[0], 'spacingY': spacing[1], 'spacingZ': spacing[2],
                     'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing, 'patient_id': patient_id, })

                # todo 保存结节
                if mode == '2d':
                    self.save2D(item, vol, one_nodule, spacing, [result, noduleSize])
                elif mode == '3d':
                    self.save3D(item, vol, one_nodule, spacing, [result, noduleSize])

                # todo 检查是否有遗漏
                if len(one_nodule) != 0:
                    if len(one_nodule) < 3:
                        print(f'nodule len:{len(one_nodule)},anns:{len(anns)},', item, one_nodule)
                    all_nodules.append(one_nodule)
                else:
                    print('none', item)

        if mode is None:
            """保存结节属性"""
            df = pd.DataFrame(attrs)
            df.to_csv(f'{config.csv_path}/all_luna_nodules_info.csv')
            print('Save luna annotation csv !!!')
            # todo 统计信息
            print('total', len(all_nodules))
            print(all_nodules)

    def save2D(self, name, vol, one_nodule, spacing, attrs):

        masks, bbox, _ = consensus(one_nodule, clevel=0.5, pad=[(32, 32), (32, 32), (0, 0)])
        # todo 选择最具有代表性的一张，则mask数量最多
        img, msk, _ = self.findmax(vol[bbox], masks)

        img = resample2d(img, spacing=spacing[:2])
        msk = resample2d(msk, spacing=spacing[:2], is_seg=True)

        img = img_crop_or_fill(img, 'twoD')
        mask = img_crop_or_fill(msk, 'twoD')

        while True:
            randomN = random.randint(0, 100)
            lesion_name = self.seg_path2d + f'{name}_{attrs[1]}_{attrs[0][0]}_{attrs[0][1]}_{attrs[0][2]}_{attrs[0][3]}' \
                                            f'_{attrs[0][4]}_{attrs[0][5]}_{attrs[0][6]}_{attrs[0][7]}_{attrs[0][8]}' \
                                            f'_{randomN}.npy'
            if not os.path.exists(lesion_name):
                break
        lesion = np.concatenate((img[np.newaxis, ...], mask[np.newaxis, ...]))
        np.save(lesion_name, lesion)
        print(name)

    def save3D(self, name, vol, nodules, spacing, attrs):
        # todo 求掩膜均值
        masks, bbox, _ = consensus(nodules, clevel=0.5, pad=[(32, 32), (32, 32), (32, 32)])

        imgs, masks = resample_patient(vol[bbox][np.newaxis, ...], masks[np.newaxis, ...], spacing,
                                       [1., 1., 1., ], force_separate_z=None)

        # todo 过滤插值失败的img
        imgs, masks = fliter(imgs, masks)

        imgs = img_crop_or_fill(imgs, '3d')
        masks = img_crop_or_fill(masks, '3d')

        while True:
            randomN = random.randint(0, 100)
            lesion_name = self.seg_path3d + f'{name}_{attrs[1]}_{attrs[0][0]}_{attrs[0][1]}_{attrs[0][2]}_{attrs[0][3]}' \
                                            f'_{attrs[0][4]}_{attrs[0][5]}_{attrs[0][6]}_{attrs[0][7]}_{attrs[0][8]}' \
                                            f'_{randomN}.npy'
            if not os.path.exists(lesion_name):
                break
        lesion = np.concatenate((imgs[np.newaxis, ...], masks[np.newaxis, ...]))
        np.save(lesion_name, lesion)
        print(name)

    def __init__(self, mode=None):
        super(FindLunaNodule, self).__init__()
        if mode == '-1':
            logs(f'lidc prepare {config.mode}')
        else:  # 非LIDC数据集
            logs(f'luna prepare {config.mode}')
            self.dirInit()
            if mode is not None:
                pool = Pool(multiprocessing.cpu_count())  # 开启线程池
                func = partial(self.main, mode=mode)
                N = 34  # 线程数
                _ = pool.map(func, range(N))
                pool.close()  # 关闭线程池
                pool.join()
            else:  # 统计信息
                self.main(-1, mode)


if __name__ == '__main__':
    """
    todo 1:统计结节信息   mode=None
    todo 2：保存2d最大横截面  mode=2d
    todo 3：保存3d 结节块     mode=3d
    """

    mode = '3d'
    print(mode)
    start_time = time.time()
    FindLunaNodule(mode).to('cuda:0')
    end_time = time.time()
    showTime(f'{mode} Total', start_time, end_time)
