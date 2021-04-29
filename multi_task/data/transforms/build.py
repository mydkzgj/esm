# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .sequence import build_transforms_sequence_classification

def build_transforms(dataset_type, cfg, is_train=True):
    if dataset_type == "sequence-classification":
        train_transforms = build_transforms_sequence_classification(cfg, is_train=is_train)
        val_transforms = build_transforms_sequence_classification(cfg, is_train=False)
        test_transforms = build_transforms_sequence_classification(cfg, is_train=False)
    else:
        raise Exception("Wrong Transforms Type!")

    return train_transforms, val_transforms, test_transforms


