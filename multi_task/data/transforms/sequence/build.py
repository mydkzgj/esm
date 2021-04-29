# encoding: utf-8
"""
@author:  chenjiayang
@contact: sychenjiayang@163.com
"""
from PIL import Image
import torchvision.transforms as T
from . import cla_transforms as CT


def build_transforms_sequence_classification(cfg, is_train=True):
    #crop_length = 2000  #777
    if is_train:
        transform = T.Compose([
            #CT.PaddingToMinimumLength(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.RandomCrop(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.Encode(),
            CT.ToTensor(),
        ])
    else:
        transform = T.Compose([
            #CT.PaddingToMinimumLength(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.RandomCrop(cfg.DATA.TRANSFORM.SEQUENCE_CROP_SIZE),
            #CT.Encode(),
            CT.ToTensor(),
        ])

    return transform
