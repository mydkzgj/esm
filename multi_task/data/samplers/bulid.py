# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler


def build_sampler(cfg, data_source, num_classes, set_name="train", is_train=True):
    if cfg.DATA.DATALOADER.SAMPLER == "sequential":
        sampler = SequentialSampler(data_source)
    elif cfg.DATA.DATALOADER.SAMPLER == "random":
        sampler = RandomSampler(data_source, replacement=False)
    else:
        raise Exception("Wrong Sampler Name!")
    return sampler
