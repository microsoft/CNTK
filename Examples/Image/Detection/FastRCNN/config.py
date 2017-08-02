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
__C.CNTK = edict()
__C.TRAIN = edict()

# If set to 'True' training will be skipped if a trained model exists already
__C.CNTK.MAKE_MODE = True
# set to 'True' to use determininistic algorithms
__C.CNTK.FORCE_DETERMINISTIC = False
# set to 'True' to run only a single epoch
__C.CNTK.FAST_MODE = False
# Debug parameters
__C.CNTK.DEBUG_OUTPUT = False
__C.CNTK.GRAPH_TYPE = "png" # "png" or "pdf"

# Learning parameters
__C.CNTK.L2_REG_WEIGHT = 0.0005
__C.CNTK.MOMENTUM_PER_MB = 0.9
__C.CNTK.MAX_EPOCHS = 15 # use more epochs and more ROIs (NUM_ROI_PROPOSALS) for better results
__C.CNTK.LR_FACTOR = 1.0
__C.CNTK.LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]
# The learning rate multiplier for all bias weights
__C.CNTK.BIAS_LR_MULT = 2.0

# Number of regions of interest [ROIs] proposals
__C.NUM_ROI_PROPOSALS = 500 # use 2000 or more for good results
# minimum width and height for proposals in pixels
__C.PROPOSALS_MIN_W = 20
__C.PROPOSALS_MIN_H = 20
# the minimum IoU (overlap) of a proposal to qualify for training regression targets
__C.BBOX_THRESH = 0.5

# Normalize the targets using "precomputed" (or made up) means and stdevs
__C.BBOX_NORMALIZE_TARGETS = True
__C.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Maximum number of ground truth annotations per image
__C.INPUT_ROIS_PER_IMAGE = 50
__C.IMAGE_WIDTH = 850
__C.IMAGE_HEIGHT = 850

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = False
# If set to 'True' conv layers weights from the base model will be trained, too
__C.TRAIN_CONV_LAYERS = True
# Sigma parameter for smooth L1 loss in the RPN and the detector (DET)
__C.SIGMA_DET_L1 = 1.0

# NMS threshold used to discard overlapping predicted bounding boxes
__C.RESULTS_NMS_THRESHOLD = 0.5
# all bounding boxes with a score lower than this threshold will be considered background
__C.RESULTS_NMS_CONF_THRESHOLD = 0.0

# Enable plotting of results generally / also plot background boxes / also plot unregressed boxes
__C.VISUALIZE_RESULTS = True
__C.DRAW_NEGATIVE_ROIS = False
__C.DRAW_UNREGRESSED_ROIS = False
# only for plotting results: boxes with a score lower than this threshold will be considered background
__C.RESULTS_BGR_PLOT_THRESHOLD = 0.1


# For reproducibility
__C.RND_SEED = 3

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = False

# Default GPU device id
__C.GPU_ID = 0
