# encoding: utf-8
"""
@author: Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import logging
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

import torch
import torch.nn as nn

import torchvision

from ignite.engine import Engine, Events
from ignite.handlers import Timer  # ModelCheckpoint,  #自己编写一下ModelCheckpoint，使其可以设置初始轮数
from utils.checkpoint import ModelCheckpoint

from ignite.metrics import RunningAverage
from ignite.metrics import MeanSquaredError
from ignite.metrics import Accuracy
from ignite.metrics import Precision

from engine.evaluator import do_inference

from torch.utils.tensorboard import SummaryWriter

from metric import *

global ITER
ITER = 0


def create_supervised_trainer(model, optimizer, metrics, loss_fn, accumulation_steps=1, device=None,):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)   #however, some attributes of original model do not pass down, so it can not be used
        model.to(device)
        # optimizer加载进来的是cpu类型，需要手动转成gpu。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def _update(engine, batch):
        """
        label can be scalar or vector
        binary-multi-label must be comprised of 0's and 1's
        """
        model.eval()
        #model.backbone.eval()
        #model.ct_predictor.train()
        #model.module.backbone.eval()
        #model.module.ct_predictor.train()
        #model.train()

        # fetch data
        tokens, gt_contacts, labels, strs, img_paths = batch   # corresponding to collate_fn

        # place data in CUDA
        tokens = tokens.to(device) if torch.cuda.device_count() >= 1 and tokens is not None else tokens
        #"""
        if torch.cuda.device_count() >= 1 and gt_contacts is not None:
            if isinstance(gt_contacts, list):
                gt_contacts = [gt_contact.to(device) for gt_contact in gt_contacts]
            else:
                gt_contacts.to(device)
        #"""

        #labels = labels.to(device) if torch.cuda.device_count() >= 1 and labels is not None else labels
        #bm_labels = torch.nn.functional.one_hot(labels, model.num_classes).float()

        # forward propagation
        results = model(tokens, need_head_weights=True)
        pd_contacts = results["contacts"]
        #gt_contacts = torch.rand_like(pd_contacts)

        # generate logits & label for L/k top accuracy
        rg_logits, rg_labels = model.contact_to_rg_vectors(pd_contacts, gt_contacts)

        # compute losses (dict)
        losses = loss_fn(pd_contacts=pd_contacts, gt_contacts=gt_contacts)
        weight = {"contact_prediction_loss":1, "secondary_structure_prediction_loss":1}
        loss = 0
        for lossKey in losses.keys():
            loss += losses[lossKey] * weight[lossKey]
        loss = loss/engine.state.accumulation_steps

        # backward propagation
        loss.backward()

        # parameter optimization
        if engine.state.iteration % engine.state.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return {"rg_logits": rg_logits, "rg_labels": rg_labels, "losses":losses, "total_loss": loss.item()}

    engine = Engine(_update)
    engine.state.accumulation_steps = accumulation_steps

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizer,
        scheduler,
        loss_fn,
        start_epoch,
):
    # 1.Load parameters from cfg
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.SOLVER.OUTPUT_DIR
    device = cfg.MODEL.DEVICE

    # 2.Recording tools setup
    # (1) Logger
    logger = logging.getLogger("classification.train")   # corresponding to logger("classification")
    logger.info("Start training")

    # (2) TensorBoard SummaryWriter
    # save progress
    writer_train = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/")
    writer_val = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/val")
    # save graph
    writer_graph = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/graph")

    inputshape = None
    try:
        data = next(iter(train_loader))
        input = data[0]
        #inputshape = (input.shape[1], input.shape[2], input.shape[3]) if len(input.shape)==4 else (input.shape[1], input.shape[2])
        inputshape = [input.shape[i] for i in range(1, len(input.shape))]
        """
        grid = torchvision.utils.make_grid(input)
        writer_graph.add_image('images', grid, 0)
        writer_graph.add_graph(model, input)
        writer_graph.flush()
        """
    except Exception as e:
        print("Failed to save model graph: {}".format(e))

    # 3.Create engine
    # metrics relevant to training
    metrics_train = {
        "avg_total_loss": RunningAverage(output_transform=lambda x: x["total_loss"]),
        "mse": RunningAverage(MeanSquaredError(output_transform=lambda x: (x["rg_logits"], x["rg_labels"]))),
    }

    # add seperate metrics
    lossKeys = cfg.LOSS.TYPE.split(" ")
    if "counts_regression_loss" in lossKeys:
        lossKeys.append("counts_classification_loss")

    for lossName in lossKeys:
        #"""
        if lossName == "contact_prediction_loss":
            metrics_train["AVG-" + "contact_prediction_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["contact_prediction_loss"])
        elif lossName == "secondary_structure_prediction_loss":
            metrics_train["AVG-" + "secondary_structure_prediction_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["secondary_structure_prediction_loss"])
        else:
            raise Exception('expected METRIC_LOSS_TYPE should not be {}'.format(cfg.LOSS.TYPE))

    # create engine with metrics attached
    trainer = create_supervised_trainer(model, optimizer, metrics_train, loss_fn, device=device, )

    # attach checkpointer & timer to the engine
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.BACKBONE_NAME, checkpoint_period, n_saved=300, require_empty=False, start_step=start_epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})

    #checkpointer_save_graph = ModelCheckpoint(output_dir, cfg.MODEL.BACKBONE_NAME, checkpoint_period, n_saved=300, require_empty=False, start_step=-1)
    #trainer.add_event_handler(Events.STARTED, checkpointer_save_graph, {'model': model, 'optimizer': optimizers[0]})

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # 4.Other event handlers
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        engine.state.iteration = engine.state.iteration + start_epoch * len(train_loader)

        logger.info("Model:{}".format(model))
        print("Input Shape: {}".format(inputshape))
        #inputshape = (cfg.DATA.TRANSFORM.CHANNEL, cfg.DATA.TRANSFORM.SIZE[0], cfg.DATA.TRANSFORM.SIZE[1])
        #logger.info("Model:{}".format(model.count_param(input_shape=inputshape)))

        #metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn, plotFlag=False)

    @trainer.on(Events.EPOCH_COMPLETED)  # 注意，在pytorch1.2里面 scheduler.steo()应该放到 optimizer.step()之后
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % (log_period * engine.state.accumulation_steps) == 0:
            step = engine.state.iteration

            # 1.Tensorboard Summary
            # loss (vector)
            avg_losses = {}
            for lossName in lossKeys:
                avg_losses[lossName] = (float("{:.3f}".format(engine.state.metrics["AVG-" + lossName])))
                writer_train.add_scalar("Loss/" + lossName, avg_losses[lossName], step)
                writer_train.flush()

            # other scalars
            scalar_list = ["mse", "avg_total_loss"]
            for scalar in scalar_list:
                writer_train.add_scalar("Train/" + scalar, engine.state.metrics[scalar], step)
                writer_train.flush()

            # learning rate
            writer_train.add_scalar("Train/" + "LearningRate", scheduler.get_lr()[0], step)
            writer_train.flush()

            # 2.logger
            logger.info("Epoch[{}] Iteration[{}/{}] ATLoss: {:.3f}, Avg_Loss: {}, MSE: {:.3f}, Base Lr: {:.2e}, step: {}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_total_loss'], avg_losses,
                                engine.state.metrics['mse'],
                                scheduler.get_lr()[0], step))

        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn, target_set_name="valid", plotFlag=False) #不进行绘制

            step = engine.state.iteration

            writer_val.add_scalar("MSE", metrics['mse'], step)
            writer_val.flush()


    # 5.launch engine
    trainer.run(train_loader, max_epochs=epochs)
    writer_train.close()
    writer_val.close()

