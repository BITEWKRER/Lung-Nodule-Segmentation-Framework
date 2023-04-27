# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

# todo 数据集路径
from torch import nn


class GC(nn.Module):
    version = None

    basePath = None
    basePathV2 = None

    # todo 数据集路径,npy 格式
    seg_path_luna_3d = None
    seg_path_luna_2d = None
    seg_path_lidc_3d = None
    seg_path_lidc_2d = None

    """模型参数路径"""
    pth_luna_path = None
    pth_lidc_path = None
    """日志路径"""
    xml_path = None
    csv_path = None

    # todo 训练、评估配置
    dataset = 'luna'
    csv_name = 'debug'  # 日志、csv 保存的文件名,
    mode = '3d'  # 训练类型
    train = False  # 控制日志文件名称
    sup = False  # 深监督学习
    """超参数"""
    device = None
    num_worker = 4
    epochs = 600
    show = False
    val_domain = 0.2
    train_domain = 0.8
    lr = 3e-4
    earlyEP = 10
    k_fold = 5
    train_batch_size = 8
    val_and_test_batch_size = 1
    optimizer = 'adam'
    model_name = ''
    loss_name = ''
    lost_loss = False
    # todo focal loss
    alpha = 0.25
    gamma = 2
    # todo input size
    input_size_2d = 64
    input_size_3d = 64

    def __init__(self, train=False, dataset='luna', log_name='debug', mode='3d', device='cuda:0', server=''):
        super(GC, self).__init__()
        self.sup = sup
        self.device = device
        self.LossV = LossV
        self.FileV = FileV
        self.MetricsV = MetricsV
        self.version = pathV
        self.train = train
        self.dataset = dataset
        self.log_name = log_name
        self.mode = mode
        self.server = server
        self.basePath = ''  # project path

        self.SetDatasetPath()
        self.SetPthPath()

        self.xml_path = f'{self.basePath}/LIDCXML/lidcxml/'
        self.csv_path = f'{self.basePath}/LIDCXML/annos/'
        self.pred_path = f'{self.basePath}/predict'

    def SetDatasetPath(self):
        self.seg_path_luna_3d = f'{self.basePath}/$segmentation/seg_luna_3d/'
        self.seg_path_luna_2d = f'{self.basePath}/$segmentation/seg_luna_2d/'
        self.seg_path_lidc_3d = f'{self.basePath}/$segmentation/seg_lidc_3d/'
        self.seg_path_lidc_2d = f'{self.basePath}/$segmentation/seg_lidc_2d/'

    def SetPthPath(self, ):
        self.pth_luna_path = f'{self.basePath}/pth_luna/'
        self.pth_lidc_path = f'{self.basePath}/pth_lidc/'


"""  
添加属性后需要在trainBase进行添加 
"""
train = False
dataset = 'lidc'
log_name = 'eva'
mode = '3d'  # 2d,3d
device = 'cuda:0'  #
server = ' '  #

config = GC(train=train, dataset=dataset, log_name=log_name, mode=mode, device=device, server=server)
