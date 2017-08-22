# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
__C.DATA = edict()
cfg = __C

# data set config
__C.DATA.DATASET = "Pascal"
__C.DATA.MAP_FILE_PATH = "../../DataSets/Pascal/mappings"
__C.DATA.CLASS_MAP_FILE = "class_map.txt"
__C.DATA.TRAIN_MAP_FILE = "trainval2007.txt"
__C.DATA.TRAIN_ROI_FILE = "trainval2007_rois_abs-xyxy_noPad_skipDif.txt"
__C.DATA.TEST_MAP_FILE = "test2007.txt"
__C.DATA.TEST_ROI_FILE = "test2007_rois_abs-xyxy_noPad_skipDif.txt"
__C.DATA.NUM_TRAIN_IMAGES = 5010
__C.DATA.NUM_TEST_IMAGES = 4952
__C.DATA.PROPOSAL_LAYER_SCALES = [8, 16, 32]

__C.DATA.TRAIN_PRECOMPUTED_PROPOSALS_FILE = "trainval2007_proposals.txt"
__C.DATA.TEST_PRECOMPUTED_PROPOSALS_FILE = "test2007_proposals.txt"
