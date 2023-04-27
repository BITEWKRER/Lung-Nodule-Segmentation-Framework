# --coding:utf-8 --
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from configs import config
from utils.evaluateBase import evaluateBase
from utils.helper import getAllAttrs


class evaluateLuna(evaluateBase):

    def __init__(self, model_lists, labels):
        super(evaluateLuna, self).__init__(model_lists)
        self.pth_path = config.pth_luna_path
        self.dataset = 'luna'
        if self.mode == '2d':
            self.seg_path = config.seg_path_luna_2d
        else:
            self.seg_path = config.seg_path_luna_3d

        self.run(labels)


if __name__ == '__main__':
    config.train = False

    """
    cmd 命令
    conda activate jwj
    cd /zsm/jwj/baseExpV3/
    cd /zljteam/jwj/baseExpV3/
    nohup python evaluateLuna.py >/dev/null 2>&1 &
    """
    # todo 2d 82983
    # todo 3d 154913
    loss_lists = ['dice', 'bce', 'focal']  #
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'transbts', 'wingsnet', 'unetr', ]

    # model3d = ['pcamnet', 'asa', 'vtunet', ]
    mode = config.mode
    evaluateLuna(model3d, None, ).to(config.device)  # 整体评估，读入全部数据

    # for labels in getAllAttrs(True).values():  # todo 分项评估
    #     evaluateLuna(model3d, labels).to(config.device)
