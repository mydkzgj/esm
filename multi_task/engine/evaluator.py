# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import logging

import os

import torch
import torch.nn as nn
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from ignite.metrics import ConfusionMatrix
from ignite.metrics import MeanSquaredError

from sklearn.metrics import roc_curve, auc

import numpy as np


def create_supervised_evaluator(model, metrics, loss_fn, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # fetch data
            tokens, gt_contacts, labels, strs, img_paths = batch   # corresponding to collate_fn

            # place data in CUDA
            tokens = tokens.to(device) if torch.cuda.device_count() >= 1 and tokens is not None else tokens
            # """
            if torch.cuda.device_count() >= 1 and gt_contacts is not None:
                if isinstance(gt_contacts, list):
                    gt_contacts = [gt_contact.to(device) for gt_contact in gt_contacts]
                else:
                    gt_contacts.to(device)
            # """
            # labels = labels.to(device) if torch.cuda.device_count() >= 1 and labels is not None else labels
            # bm_labels = torch.nn.functional.one_hot(labels, model.num_classes).float()

            # forward propagation
            results = model(tokens, need_head_weights=True)
            pd_contacts = results["contacts"]
            # gt_contacts = torch.rand_like(pd_contacts)

            # generate logits & label for L/k top accuracy
            rg_logits, rg_labels = model.contact_to_rg_vectors(pd_contacts, gt_contacts)

            return {"rg_logits": rg_logits, "rg_labels": rg_labels}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_inference(
        cfg,
        model,
        test_loader,
        classes_list,
        loss_fn,
        target_set_name="test",
        plotFlag=False
):
    num_classes = len(classes_list) if classes_list is not None else 0
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("classification.inference")
    logging._warn_preinit_stderr = 0
    logger.info("Enter inferencing for {} set".format(target_set_name))

    metrics_eval = {
        "mse": MeanSquaredError(output_transform=lambda x: (x["rg_logits"], x["rg_labels"])),
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    metrics = dict()

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        logger.info("Test Results")
        if engine.state.metrics.get("mse") != None:
            mse = engine.state.metrics["mse"]
            logger.info("MSE: {:.3f}".format(mse))
            metrics["mse"] = mse

    evaluator.run(test_loader)

    return metrics



