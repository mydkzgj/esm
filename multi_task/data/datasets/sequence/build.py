# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
"""
import os
from data.transforms import *

from .general_sequence import General_Sequence
from .cath import CATH


def build_sequence_datasets(dataset_name, dataset_type, cfg, is_train):
    train_transforms = build_transforms(dataset_type, cfg, is_train=is_train)
    val_transforms = build_transforms(dataset_type, cfg, is_train=False)
    test_transforms = build_transforms(dataset_type, cfg, is_train=False)

    root_path = cfg.DATA.DATASETS.ROOT_DIR

    if dataset_name == "general-sequence":
        root_path = os.path.join(root_path, "DATABASE", "General-Sequence")
        train_set = General_Sequence(root=root_path, set_name="train", transform=train_transforms)
        val_set = General_Sequence(root=root_path, set_name="valid", transform=val_transforms)
        test_set = General_Sequence(root=root_path, set_name="test", transform=test_transforms)
        classes_list = None
    elif dataset_name == "cath":
        #root_path = os.path.join(root_path, "DATABASE", "General-Sequence")
        train_set = CATH(root=root_path, set_name="train", transform=train_transforms)
        val_set = CATH(root=root_path, set_name="valid", transform=val_transforms)
        test_set = CATH(root=root_path, set_name="test", transform=test_transforms)
        classes_list = None
    else:
        raise Exception("Can not build sequence dataset: {}".format(dataset_name))

    return train_set, val_set, test_set, classes_list