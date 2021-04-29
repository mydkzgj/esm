# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
import numpy as np
import random

sys.path.append('.')

from data import make_data_loader

from model import build_model
from loss import build_loss
from engine import do_inference

from config import cfg
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def eval(cfg, target_set_name="test"):
    # prepare dataset
    train_loader, val_loader, test_loader, classes_list = make_data_loader(cfg, for_train=False)
    num_classes = len(classes_list)

    # build model and load parameter
    model = build_model(cfg)
    model.load_param("Overall", cfg.TEST.WEIGHT)

    # build loss function
    loss_func, loss_class = build_loss(cfg)
    print('Eval with losses:', cfg.LOSS.TYPE)

    # input data_loader
    if target_set_name == "train":
        input_data_loader = train_loader
    elif target_set_name == "valid":
        input_data_loader = val_loader
    elif target_set_name == "test":
        input_data_loader = test_loader
    else:
        raise Exception("Wrong Dataset Name!")

    # build and launch engine for evaluation
    metrics = do_inference(cfg,
                           model,
                           input_data_loader,
                           classes_list,
                           loss_func,
                           target_set_name=target_set_name,
                           plotFlag=True)

    # logging with tensorboard summaryWriter
    model_epoch = cfg.TEST.WEIGHT.split('/')[-1].split('.')[0].split('_')[-1]
    model_iteration = len(train_loader) * int(model_epoch) if model_epoch.isdigit() == True else 0

    writer_test = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/eval_" + target_set_name)

    writer_test.add_scalar("MSE", metrics["mse"], model_iteration)

    writer_test.close()


def main():
    parser = argparse.ArgumentParser(description="Classification Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--target_set", default="", help="name of target dataset: train, valid, test, all", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("classification", output_dir, "eval_on_{}".format(args.target_set), 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join("%s"%i for i in cfg.MODEL.DEVICE_ID)   # int tuple -> str # cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    logger.info("Eval on the {} dataset".format(args.target_set))
    if args.target_set == "train" or args.target_set == "valid" or args.target_set == "test":
        eval(cfg, args.target_set)
    elif args.target_set == "all":
        eval(cfg, "train")
        eval(cfg, "valid")
        eval(cfg, "test")
    else:
        raise Exception("Wrong dataset name with {}".format(args.dataset_name))


if __name__ == '__main__':
    seed_torch(2018)
    main()
