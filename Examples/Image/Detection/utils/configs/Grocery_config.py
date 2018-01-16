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
__C.DATA.DATASET = "Grocery"
__C.DATA.MAP_FILE_PATH = "../../DataSets/Grocery"
__C.DATA.CLASS_MAP_FILE = "class_map.txt"
__C.DATA.TRAIN_MAP_FILE = "train_img_file.txt"
__C.DATA.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.DATA.TEST_MAP_FILE = "test_img_file.txt"
__C.DATA.TEST_ROI_FILE = "test_roi_file.txt"
__C.DATA.NUM_TRAIN_IMAGES = 20
__C.DATA.NUM_TEST_IMAGES = 5
__C.DATA.PROPOSAL_LAYER_SCALES = [4, 8, 12]

# overwriting proposal parameters for Fast R-CNN
# minimum relative width/height of an ROI
__C.roi_min_side_rel = 0.04
# maximum relative width/height of an ROI
__C.roi_max_side_rel = 0.4
# minimum relative area of an ROI
__C.roi_min_area_rel = 2 * __C.roi_min_side_rel * __C.roi_min_side_rel
# maximum relative area of an ROI
__C.roi_max_area_rel = 0.33 * __C.roi_max_side_rel * __C.roi_max_side_rel
# maximum aspect ratio of an ROI vertically and horizontally
__C.roi_max_aspect_ratio = 4.0

# For this data set use the following lr factor for Fast R-CNN:
# __C.CNTK.LR_FACTOR = 10.0
