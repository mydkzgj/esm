# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

"""
Solver contain optimizer and lr_scheduler, which correspond with torch.optim.Optimizer & torch.optim.lr_scheduler.
By the way, lr_scheduler is dependent on optimizer.
"""

from .optimizer import build_optimizer
from .lr_scheduler import build_scheduler


