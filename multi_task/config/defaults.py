from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1.Data General Setting. Can be replaced in respective sets
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.DATA = CN()
# -----------------------------------------------------------------------------
# Data.Dataset
# -----------------------------------------------------------------------------
_C.DATA.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATA.DATASETS.NAMES = ('none')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATA.DATASETS.ROOT_DIR = ('./data')
# -----------------------------------------------------------------------------
# Data.DataLoader
# -----------------------------------------------------------------------------
_C.DATA.DATALOADER = CN()
# Number of data loading threads
_C.DATA.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATA.DATALOADER.SAMPLER = "random"  #"sequential", "random", "weighted_random", "class_balance_random"  #'softmax_rank'
# Number of samples per batch during test
_C.DATA.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of samples per batch during test
_C.DATA.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of samples per batch
_C.DATA.DATALOADER.IMS_PER_BATCH = _C.DATA.DATALOADER.CATEGORIES_PER_BATCH * _C.DATA.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
# -----------------------------------------------------------------------------
# Data.TRANSFORM
# -----------------------------------------------------------------------------
_C.DATA.TRANSFORM = CN()
# Padding To Square
_C.DATA.TRANSFORM.PADDING_TO_SQUARE_MODE = "none"   #"none", "constant", "edge", "reflect"(mirror), "symmetric"
# Channel of the image
_C.DATA.TRANSFORM.CHANNEL = 3
# Size of the image during training
_C.DATA.TRANSFORM.SIZE = [224, 224]
# 掩膜缩放比例
_C.DATA.TRANSFORM.MASK_SIZE_RATIO = 1  #4
# Values to be used for image normalization
_C.DATA.TRANSFORM.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.DATA.TRANSFORM.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.DATA.TRANSFORM.PADDING = 10
# Random probability for image horizontal flip
_C.DATA.TRANSFORM.PROB = 0.5
# Random probability for random erasing
_C.DATA.TRANSFORM.RE_PROB = 0.5

# Sequence Crop Length
_C.DATA.TRANSFORM.SEQUENCE_CROP_SIZE = 2000


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.Model General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2-1 Basic Config
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = (0, 1)  #it has bug with “both str and int convert to int, so use tuple instead which seems to turn into remove brackets automatically”
# Name of backbone
_C.MODEL.BACKBONE_NAME = 'densenet121'
# if backbone with classifier itself: True or False
_C.MODEL.BACKBONE_WITH_CLASSIFIER = 0
# if with classifier, this parameter can be used or not.
_C.MODEL.BWC_NUM_CLASSES = 0
# input type of model: "single-instance" or "multi-instance"
_C.MODEL.BACKBONE_INPUT_TYPE = "single-instance"

# Name of Contact Predictor
_C.MODEL.CONTACT_PREDICTOR_NAME = "none"

# Name of Secondary Structure Predictor
_C.MODEL.SECONDARY_STRUCTURE_PREDICTOR_NAME = "none"

# if setting bias-free for model ## actually implemented in optimizer (zero initialization without subsequent optimization）
_C.MODEL.BIAS_FREE = 0

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3.LOSS General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#LOSS configuration
_C.LOSS = CN()
# The loss type of metric loss
# options:'cranked_loss','ranked_loss',
_C.LOSS.TYPE = 'cross_entropy_loss'

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4.Solver
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# Path to checkpoint and saved log of trained model
_C.SOLVER.OUTPUT_DIR = "work_space"
# -----------------------------------------------------------------------------
# OPTIMIZER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.OPTIMIZER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER.NAME = "Adam"
# Momentum
_C.SOLVER.OPTIMIZER.MOMENTUM = 0.9
# Settings of weight decay
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS = 0.
# accumulation_steps (only used in training)
_C.SOLVER.OPTIMIZER.ACCUMULATION_STEP = 1
# -----------------------------------------------------------------------------
# SCHEDULER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.SCHEDULER = CN()
# Base learning rate
_C.SOLVER.SCHEDULER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.SCHEDULER.BIAS_LR_FACTOR = 2
# decay rate of learning rate
_C.SOLVER.SCHEDULER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.SCHEDULER.STEPS = (30, 55)
# warm up factor
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.SCHEDULER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.SCHEDULER.WARMUP_METHOD = "linear"
# center loss lr
_C.SOLVER.SCHEDULER.LOSS_LR = 0.05
# if train from the head
_C.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD = 1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Train Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Path to trained model
_C.TRAIN.WEIGHT = ""
# -----------------------------------------------------------------------------
# Train Transform
# -----------------------------------------------------------------------------
_C.TRAIN.TRANSFORM = CN()

# -----------------------------------------------------------------------------
# Train Dataloader
# -----------------------------------------------------------------------------
_C.TRAIN.DATALOADER = CN()
# Number of categories per batch
_C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.TRAIN.DATALOADER.IMS_PER_BATCH = _C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * _C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH


# -----------------------------------------------------------------------------
# Train Trick
# -----------------------------------------------------------------------------
_C.TRAIN.TRICK = CN()
# Path to pretrained model of backbone
_C.TRAIN.TRICK.PRETRAIN_PATH = r'PATH\unknown.pth'
# Pretrained or not
_C.TRAIN.TRICK.PRETRAINED = 1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Val Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.VAL = CN()
# Path to trained model
_C.VAL.WEIGHT = ""
# -----------------------------------------------------------------------------
# Val Transform
# -----------------------------------------------------------------------------
#_C.VAL.TRANSFORM = CN()
# Random probability for image horizontal flip
#_C.VAL.TRANSFORM.PROB = 0.5
# Random probability for random erasing
#_C.VAL.TRANSFORM.RE_PROB = 0.5
# -----------------------------------------------------------------------------
# Val Dataloader
# -----------------------------------------------------------------------------
_C.VAL.DATALOADER = CN()
# Number of categories per batch
_C.VAL.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.VAL.DATALOADER.IMS_PER_BATCH = _C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * _C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Test Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Path to trained model
_C.TEST.WEIGHT = ""
# -----------------------------------------------------------------------------
# Test Transform
# -----------------------------------------------------------------------------
#_C.TEST.TRANSFORM = CN()
# Random probability for image horizontal flip
#_C.TEST.TRANSFORM.PROB = 0.5
# Random probability for random erasing
#_C.TEST.TRANSFORM.RE_PROB = 0.5
# -----------------------------------------------------------------------------
# Test Dataloader
# -----------------------------------------------------------------------------
_C.TEST.DATALOADER = CN()
# Number of categories per batch
_C.TEST.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.TEST.DATALOADER.IMS_PER_BATCH = _C.TEST.DATALOADER.CATEGORIES_PER_BATCH * _C.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH

