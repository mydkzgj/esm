# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import argparse
import os
import sys
import torch
from torch.backends import cudnn
import numpy as np
import random

sys.path.append('.')
sys.path.append('..')

from data import make_data_loader

from model import build_model
from loss import build_loss
from solver import build_optimizer, build_scheduler
from engine import do_train

from config import cfg
from utils.logger import setup_logger


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(cfg):
    # prepare dataset
    train_loader, val_loader, test_loader, classes_list = make_data_loader(cfg, for_train=True)

    # build model and load parameter
    model = build_model(cfg)
    if cfg.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD == True:
        if cfg.TRAIN.TRICK.PRETRAINED == True:
            model.load_param("Base", cfg.TRAIN.TRICK.PRETRAIN_PATH)
    else:
        if cfg.TRAIN.TRICK.PRETRAINED == True:
            model.load_param("Overall", cfg.TRAIN.TRICK.PRETRAIN_PATH)

    train_loader.dataset.batch_converter = model.backbone_batch_converter
    val_loader.dataset.batch_converter = model.backbone_batch_converter
    test_loader.dataset.batch_converter = model.backbone_batch_converter

    # build loss function
    loss_func, loss_class = build_loss(cfg)
    print('Train with losses:', cfg.LOSS.TYPE)

    # build optimizer （based on model）
    optimizer = build_optimizer(cfg, model, bias_free=cfg.MODEL.BIAS_FREE)  #loss里也可能有参数
    print("Model Bias-Free:{}".format(cfg.MODEL.BIAS_FREE))
    print('Train with the optimizer type is', cfg.SOLVER.OPTIMIZER.NAME)

    # build scheduler （based on optimizer）
    scheduler, start_epoch = build_scheduler(cfg, optimizer)

    # build and launch engine for training
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizer,
        scheduler,
        loss_func,
        start_epoch,
    )


def main():
    #解析命令行参数,详见argparse模块
    parser = argparse.ArgumentParser(description="Classification Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)    #nargs=argparse.REMAINDER是指所有剩余的参数均转化为一个列表赋值给此项

    args = parser.parse_args()
     
    #os.environ()是python用来获取系统相关信息的。如environ[‘HOME’]就代表了当前这个用户的主目录
    ## WORLD_SIZE 由torch.distributed.launch.py产生 具体数值为 nproc_per_node*node(主机数，这里为1)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    #此处是指如果有类似yaml重新赋值参数的文件在的话会把它读进来。这也是rbgirshick/yacs模块的优势所在——参数与代码分离
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()          #最终要freeze一下，prevent further modification，也就是参数设置在这一步就完成了，后面都不能再改变了

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #logger主要用于输出运行日志，相比print有一定优势。
    logger = setup_logger("classification", output_dir, "training", 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    # print the configuration file
    '''
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    #'''
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join("%s"%i for i in cfg.MODEL.DEVICE_ID)   # int tuple -> str # cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train(cfg)


if __name__ == '__main__':
    seed_torch(2018)
    main()
