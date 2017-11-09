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
# set to 'True' to use deterministic algorithms
__C.CNTK.FORCE_DETERMINISTIC = False
# set to 'True' to run only a single epoch
__C.CNTK.FAST_MODE = False
# Debug parameters
__C.CNTK.DEBUG_OUTPUT = False
__C.CNTK.GRAPH_TYPE = "png" # "png" or "pdf"

# Learning parameters
__C.CNTK.L2_REG_WEIGHT = 0.0005
__C.CNTK.MOMENTUM_PER_MB = 0.9
__C.CNTK.MAX_EPOCHS = 20
__C.CNTK.LR_FACTOR = 10.0 # 10.0 is used for the Grocery example data. Start with 1.0 for other data or when using VGG16 as base model.
__C.CNTK.LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]
# The learning rate multiplier for all bias weights
__C.CNTK.BIAS_LR_MULT = 2.0

# Number of regions of interest [ROIs] proposals
__C.NUM_ROI_PROPOSALS = 200 # use 2000 or more for good results
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
__C.TRAIN.USE_FLIPPED = True
# If set to 'True' conv layers weights from the base model will be trained, too
__C.TRAIN_CONV_LAYERS = True
# Sigma parameter for smooth L1 loss in the RPN and the detector (DET)
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


# If set to True the following two parameters need to point to the corresponding files that contain the proposals:
# __C.DATA.TRAIN_PRECOMPUTED_PROPOSALS_FILE
# __C.DATA.TEST_PRECOMPUTED_PROPOSALS_FILE
__C.USE_PRECOMPUTED_PROPOSALS = False

# roi proposal parameters for selective search, grid and filtering
# The first three parameters are for dlib's selective search. For details see
# http://dlib.net/dlib/image_transforms/segment_image_abstract.h.html#find_candidate_object_locations
#
# The basic segmentation is performed kvals.size() times. The k parameter is set (from, to, step_size)
__C.roi_ss_kvals = (10, 500, 5)
# When doing the basic segmentations prior to any box merging, all
# rectangles that have an area < min_size are discarded. Therefore, all outputs and
# subsequent merged rectangles are built out of rectangles that contain at
# least min_size pixels. Note that setting min_size to a smaller value than
# you might otherwise be interested in using can be useful since it allows a
# larger number of possible merged boxes to be created
__C.roi_ss_min_size = 9
# There are max_merging_iterations rounds of neighboring blob merging.
# Therefore, this parameter has some effect on the number of output rectangles
# you get, with larger values of the parameter giving more output rectangles.
# Hint: set __C.CNTK.DEBUG_OUTPUT=True to see the number of ROIs from selective search
__C.roi_ss_mm_iterations = 30
#
# image size used for ROI generation
__C.roi_ss_img_size = 200
# minimum relative width/height of an ROI
__C.roi_min_side_rel = 0.01
# maximum relative width/height of an ROI
__C.roi_max_side_rel = 1.0
# minimum relative area of an ROI
__C.roi_min_area_rel = 0.0001
# maximum relative area of an ROI
__C.roi_max_area_rel = 0.9
# maximum aspect ratio of an ROI vertically and horizontally
__C.roi_max_aspect_ratio = 4.0
# aspect ratios of ROIs for uniform grid ROIs
__C.roi_grid_aspect_ratios = [1.0, 2.0, 0.5]

# For reproducibility
__C.RND_SEED = 3

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = False

# Default GPU device id
__C.GPU_ID = 0
