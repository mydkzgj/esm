# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

def build_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD == True:   # from scratch
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.SCHEDULER.STEPS,
                                      cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
                                      cfg.SOLVER.SCHEDULER.WARMUP_METHOD)

    else:   # from the last checkpoint
        start_epoch = eval(cfg.TRAIN.TRICK.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.SCHEDULER.STEPS,
                                      cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
                                      cfg.SOLVER.SCHEDULER.WARMUP_METHOD, )
                                      # start_epoch)   # KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
        for i in range(start_epoch):
            scheduler.step()
    return scheduler, start_epoch



from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
