# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline

def build_model(cfg):
    model = Baseline(
        backbone_name=cfg.MODEL.BACKBONE_NAME,
        contact_predictor_name=cfg.MODEL.CONTACT_PREDICTOR_NAME,
        secondary_structure_predictor_name=cfg.MODEL.SECONDARY_STRUCTURE_PREDICTOR_NAME,
    )
    return model
