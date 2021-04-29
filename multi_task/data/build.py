# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

from .datasets import build_datasets
from .samplers import build_sampler
from .collate_function import build_collate_fn

from torch.utils.data import DataLoader


def find_dataset_type(dataset_name):
    # torchvision + user-defined
    type_dict = {
        "sequence-classification": ["amr-genome", "general-sequence", "cath"],
    }

    dataset_type = None
    for key in type_dict.keys():
        if dataset_name in type_dict[key]:
            dataset_type = key
            break

    if dataset_type == None:
        raise Exception("Can not find matched dataset name!")

    return dataset_type


def make_data_loader(cfg, for_train):
    if cfg.DATA.DATASETS.NAMES == "none":
        return None, None, None, None
    dataset_names = cfg.DATA.DATASETS.NAMES.split(" ")
    if len(dataset_names) == 1:
        dataset_name = dataset_names[0]
    else:
        raise Exception

    dataset_type = find_dataset_type(dataset_name)

    train_set, val_set, test_set, classes_list = build_datasets(dataset_name, dataset_type, cfg, for_train)

    num_classes = len(classes_list) if classes_list is not None else 0
    train_sampler = build_sampler(cfg, train_set, num_classes, set_name="train", is_train=for_train)
    val_sampler = build_sampler(cfg, val_set, num_classes, set_name="val", is_train=False)
    test_sampler = build_sampler(cfg, test_set, num_classes, set_name="test", is_train=False)

    collate_fn = build_collate_fn(dataset_type)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS

    drop_last = False

    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, sampler=test_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )
    # notes:
    # 1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label
    return train_loader, val_loader, test_loader, classes_list
