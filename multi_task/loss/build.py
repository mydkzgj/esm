# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

# Existing Losses
from torch.nn import CrossEntropyLoss

# Custom Losses (None)
from .secondary_structure_prediction_loss import SecondaryStructurePredictionLoss
from .contact_prediction_loss import ContactPredictionLoss

def build_loss(cfg):
    """
    :param cfg: primarily base on the LOSS.TYPE of cfg.
    :return: two dicts for losses (class & function)
    """
    lossNames = cfg.LOSS.TYPE.split(" ")

    # build loss class  (most of time is useless, in case of need)
    loss_classes = {}
    for lossName in lossNames:
        if lossName == "contact_prediction_loss":
            loss_classes[lossName] = ContactPredictionLoss()
        elif lossName == "secondary_structure_prediction_loss":
            loss_classes[lossName] = SecondaryStructurePredictionLoss()
        else:
            raise Exception('Unexpected LOSS_TYPE arise: {}'.format(cfg.LOSS.TYPE))

    # build loss func
    def loss_func(pd_contacts=None, gt_contacts=None, pd_structure2s=None, gt_structure2s=None):
        losses = {}
        for lossName in lossNames:
            if lossName == "contact_prediction_loss":
                losses[lossName] = loss_classes[lossName](pd_contacts, gt_contacts)
            elif lossName == "secondary_structure_prediction_loss":
                losses[lossName] = loss_classes[lossName](pd_structure2s, gt_structure2s)
            else:
                raise Exception('Unexpected LOSS_TYPE arise: {}'.format(cfg.LOSS.TYPE))
        return losses

    return loss_func, loss_classes

