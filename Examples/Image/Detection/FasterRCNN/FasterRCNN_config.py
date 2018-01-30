# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
# `pip install easydict` if you don't have it

__C = edict()
cfg = __C

#
# CNTK parameters
#

__C.CNTK = edict()

# If set to 'True' training will be skipped if a trained model exists already
__C.CNTK.MAKE_MODE = False
# E2E or 4-stage training
__C.CNTK.TRAIN_E2E = True
# set to 'True' to use deterministic algorithms
__C.CNTK.FORCE_DETERMINISTIC = False
# set to 'True' to run only a single epoch
__C.CNTK.FAST_MODE = False
# Debug parameters
__C.CNTK.DEBUG_OUTPUT = False
__C.CNTK.GRAPH_TYPE = "png" # "png" or "pdf"
# Set to True if you want to store an eval model with native UDFs (e.g. for inference using C++ or C#)
__C.STORE_EVAL_MODEL_WITH_NATIVE_UDF = False

# Learning parameters
__C.CNTK.L2_REG_WEIGHT = 0.0005
__C.CNTK.MOMENTUM_PER_MB = 0.9
# The learning rate multiplier for all bias weights
__C.CNTK.BIAS_LR_MULT = 2.0

# E2E learning parameters
__C.CNTK.E2E_MAX_EPOCHS = 20
__C.CNTK.E2E_LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]

# 4-stage learning parameters (alternating training scheme)
__C.CNTK.RPN_EPOCHS = 16
__C.CNTK.RPN_LR_PER_SAMPLE = [0.001] * 12 + [0.0001] * 4
__C.CNTK.FRCN_EPOCHS = 8
__C.CNTK.FRCN_LR_PER_SAMPLE = [0.001] * 6 + [0.0001] * 2

# Maximum number of ground truth annotations per image
__C.INPUT_ROIS_PER_IMAGE = 50
__C.IMAGE_WIDTH = 850
__C.IMAGE_HEIGHT = 850

# Sigma parameter for smooth L1 loss in the RPN and the detector (DET)
__C.SIGMA_RPN_L1 = 3.0
__C.SIGMA_DET_L1 = 1.0

# NMS threshold used to discard overlapping predicted bounding boxes
__C.RESULTS_NMS_THRESHOLD = 0.5
# all bounding boxes with a score lower than this threshold will be considered background
__C.RESULTS_NMS_CONF_THRESHOLD = 0.0

# Enable plotting of results generally / also plot background boxes / also plot unregressed boxes
__C.VISUALIZE_RESULTS = False
__C.DRAW_NEGATIVE_ROIS = False
__C.DRAW_UNREGRESSED_ROIS = False
# only for plotting results: boxes with a score lower than this threshold will be considered background
__C.RESULTS_BGR_PLOT_THRESHOLD = 0.1

#
# Training parameters
#

__C.TRAIN = edict()

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True
# If set to 'True' conv layers weights from the base model will be trained, too
__C.TRAIN_CONV_LAYERS = True

# RPN parameters
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16

# Detector parameters
# Minibatch size (number of regions of interest [ROIs]) -- was: __C.TRAIN.BATCH_SIZE = 128
__C.NUM_ROI_PROPOSALS = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Normalize the targets using "precomputed" (or made up) means and stdevs
__C.BBOX_NORMALIZE_TARGETS = True
__C.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

#
# Testing parameters
#

__C.TEST = edict()

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16


#
# MISC
#

# For reproducibility
__C.RND_SEED = 3

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = False

# Default GPU device id
__C.GPU_ID = 0
