# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import logging

from configs import config

if config.train:
    log_name_path = f'{config.basePath}/{config.mode}_{config.dataset}_train.txt'
else:
    log_name_path = f'{config.basePath}/{config.mode}_{config.dataset}_{config.log_name}.txt'

logger = logging.getLogger('info')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_name_path)
console = logging.StreamHandler()

handler.setLevel(logging.INFO)
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


def logs(info):
    logger.info(info)
