# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import torch

def build_optimizer(cfg, model, bias_free=False, backbone_frozen=True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.SCHEDULER.BASE_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.SCHEDULER.BASE_LR * cfg.SOLVER.SCHEDULER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS

            # bias-free CJY
            if bias_free == True:
                torch.nn.init.constant_(value, 0.0)
                continue

        if backbone_frozen == True and "backbone" in key:
            #print("skip optimize {}".format(key))
            continue

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER.NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(params, momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(params)
    return optimizer



