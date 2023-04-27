# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/18 9:22
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
import sys

# # todo 无法运行cmd时，取消注释下一行
from functools import partial
from multiprocessing import Pool

sys.path.append(os.pardir)  # 环境变量

import numpy as np
import pandas as pd
import pylidc as pl

from DatasetPrepare.LunaPrepare import FindLunaNodule
from utils.logger import logs

from configs import config


class LidcPrepare(FindLunaNodule):

    def __init__(self, mode):
        super(LidcPrepare, self).__init__('-1')
        self.dataset = 'lidc'
        self.dirInit()

        pool = Pool()  # 开启线程池
        func = partial(self.main, mode=mode)
        N = 34  # 线程数
        _ = pool.map(func, range(N))
        pool.close()  # 关闭线程池
        pool.join()

    def dirInit(self):
        self.seg_path3d = config.seg_path_lidc_3d
        self.seg_path2d = config.seg_path_lidc_2d

        os.makedirs(self.seg_path2d, exist_ok=True)
        os.makedirs(self.seg_path3d, exist_ok=True)
        print(f'dir init in {self.seg_path2d},{self.seg_path3d}')

    def inList(self, data, item):
        if item in np.unique(data['seriesuid']):
            return True
        return False

    def checkFalse(self, node):
        existFalse = False
        for nod in node:  # 遍历lidc中所有的标注，查看是否出错
            if nod.subtlety in range(1, 6) and nod.internalStructure in range(1, 5) and nod.calcification in range(1, 7) \
                    and nod.sphericity in range(1, 6) and nod.margin in range(1, 6) and nod.lobulation in range(1, 6) \
                    and nod.spiculation in range(1, 6) and nod.texture in range(1, 6):
                continue
            else:
                existFalse = True
                break
        return existFalse

    def main(self, idx, mode=None):

        CTs = pl.query(pl.Scan).all()
        data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
        mhd_origin_spacing = pd.read_csv(f'{config.csv_path}/MhdOriginAndSpaving.csv', header=0)
        orgins_count = self.findAllRow(mhd_origin_spacing)
        count = self.findAllRow(data)

        attrs = []
        # 多线程并发
        size = 30
        start = size * idx
        end = start + size
        if idx == 33:
            end = None
        if idx == -1:
            start = 0
            end = None

        print(f'Thread : {idx}, start {start} to {end}，{len(CTs[start:end]) == size},{len(CTs[start:end])}')
        for ct in CTs[start:end]:
            """
            1. 如果 series_instance_uid 在luna16中，排除luna16的结节，将剩余结节进行划分
            2. 如果不在，直接聚类，输出结果
            """
            nods = ct.cluster_annotations()
            vol = ct.to_volume(verbose=False)
            vol = self.lumTrans(vol)

            spacing = ct.spacings
            slice_thickness = ct.slice_thickness
            pixel_spacing = ct.pixel_spacing
            patient_id = ct.patient_id
            item = ct.series_instance_uid

            if self.inList(data, item):  # todo luna
                print('inlist')
                citem = count[item]
                idx = orgins_count[item][0]
                mhdOrigin = np.array([mhd_origin_spacing['originX'][idx], mhd_origin_spacing['originY'][idx],
                                      mhd_origin_spacing['originZ'][idx]], dtype=np.float64)
                mhdSpacing = np.array([mhd_origin_spacing['spacingX'][idx], mhd_origin_spacing['spacingY'][idx],
                                       mhd_origin_spacing['spacingZ'][idx]], dtype=np.float64)
                # plan 1
                for node in nods:
                    pointx = []
                    pointy = []
                    pointz = []
                    for nod in node:  # todo 计算 所有注解平均值
                        pointx.append(nod.centroid[0])
                        pointy.append(nod.centroid[1])
                        pointz.append(nod.centroid[2])
                    centroid = [np.average(pointx), np.average(pointy), np.average(pointz)]
                    for row in citem:  # 统计id下的所有结节数量，这个luna数据集的判断
                        xyz = np.array([data['coordX'][row], data['coordY'][row], data['coordZ'][row]],
                                       dtype=np.float64)
                        diameater = np.floor(data['diameter_mm'][row])
                        xyz = np.round(self.worldToVoxelCoord(xyz[::-1], mhdOrigin, mhdSpacing)[::-1], 2)
                        x, y, z = xyz[0], xyz[1], xyz[2]
                        v1 = np.round(centroid[0], 2)
                        v2 = np.round(centroid[1], 2)
                        v3 = np.round(centroid[2], 2)
                        diffz = abs(z - v3)
                        diffy = abs(y - v1)
                        diffx = abs(x - v2)
                        bias = 5.
                        if diffx <= bias and diffy <= bias and diffz <= bias:
                            # 匹配成功,删除点
                            citem = [i for i in citem if i != row]
                            nods = [nod for nod in nods if nod != node]

                for node in nods:  # todo 将剩余点进行生成
                    # 检查是否标注出错，如果存在标注出错，直接排除
                    if self.checkFalse(node):
                        continue
                    result, noduleSize, avgDiameter, features, centroid = self.mix(node, None, UseLunaDiameter=False)
                    # todo 保存结节信息
                    # attrs.append(
                    #     {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1],
                    #      'centroidZ': centroid[2], 'diameter': avgDiameter, 'subtlety': features[0],
                    #      'internalStructure': features[1], 'calcification': features[2],
                    #      'sphericity': features[3],
                    #      'margin': features[4], 'lobulation': features[5], 'spiculation': features[6],
                    #      'texture': features[7], 'malignancy': features[8], 'spacingX': spacing[0],
                    #      'spacingY': spacing[1], 'spacingZ': spacing[2], 'slice_thickness': slice_thickness,
                    #      'pixel_spacing': pixel_spacing, 'patient_id': patient_id, })

                    # todo 保存结节
                    if mode == '2d':
                        self.save2D(item, vol, node, spacing, [result, noduleSize])
                    elif mode == '3d':
                        self.save3D(item, vol, node, spacing, [result, noduleSize])
            else:
                # plan 2
                # todo 保存不同掩码均值属性,获取该结节的相关属性信息
                for node in nods:
                    # 检查是否标注出错，如果存在标注出错，直接排除
                    if self.checkFalse(node):
                        continue
                    result, noduleSize, avgDiameter, features, centroid = self.mix(node, None, UseLunaDiameter=False)

                    # attrs.append(
                    #     {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1],
                    #      'centroidZ': centroid[2], 'diameter': avgDiameter, 'subtlety': features[0],
                    #      'internalStructure': features[1], 'calcification': features[2], 'sphericity': features[3],
                    #      'margin': features[4], 'lobulation': features[5], 'spiculation': features[6],
                    #      'texture': features[7], 'malignancy': features[8], 'spacingX': spacing[0],
                    #      'spacingY': spacing[1], 'spacingZ': spacing[2], 'slice_thickness': slice_thickness,
                    #      'pixel_spacing': pixel_spacing, 'patient_id': patient_id, })

                    # todo 保存结节
                    if mode == '2d':
                        self.save2D(item, vol, node, spacing, [result, noduleSize])
                    elif mode == '3d':
                        self.save3D(item, vol, node, spacing, [result, noduleSize])
        logs("end lidc")


if __name__ == '__main__':
    """
    nohup python LidcPrepare.py >/dev/null 2>&1 &
    """
    #  3d
    #  2d

    mode = config.data_type
    LidcPrepare(mode).to(config.device)
