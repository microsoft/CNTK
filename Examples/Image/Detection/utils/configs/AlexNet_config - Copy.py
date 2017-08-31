# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
__C.MODEL = edict()
cfg = __C

# model config
__C.MODEL.BASE_MODEL = "AlexNet"
if True:
    __C.MODEL.BASE_MODEL_FILE = "AlexNet_ImageNet_Caffe.model"
    __C.MODEL.IMG_PAD_COLOR = [114, 114, 114]
    __C.MODEL.FEATURE_NODE_NAME = "data" # "features"
    __C.MODEL.LAST_CONV_NODE_NAME = "relu5" # "conv5.y"
    __C.MODEL.START_TRAIN_CONV_NODE_NAME = __C.MODEL.FEATURE_NODE_NAME
    __C.MODEL.POOL_NODE_NAME = "pool5" # "pool3"
    __C.MODEL.LAST_HIDDEN_NODE_NAME = "drop7" # "h2_d"
elif False:
    __C.MODEL.BASE_MODEL_FILE = "_AlexNet.model"
    __C.MODEL.IMG_PAD_COLOR = [114, 114, 114]
    __C.MODEL.FEATURE_NODE_NAME = "features"
    __C.MODEL.LAST_CONV_NODE_NAME = "conv5.y"
    __C.MODEL.START_TRAIN_CONV_NODE_NAME = __C.MODEL.FEATURE_NODE_NAME
    __C.MODEL.POOL_NODE_NAME = "pool3"
    __C.MODEL.LAST_HIDDEN_NODE_NAME = "h2_d"
else:
    __C.MODEL.BASE_MODEL_FILE = "AlexNet_ImageNet_CNTK.model"
    __C.MODEL.IMG_PAD_COLOR = [114, 114, 114]
    __C.MODEL.FEATURE_NODE_NAME = "features"
    __C.MODEL.LAST_CONV_NODE_NAME = "z.x._.x._.x.x" # "conv5.y"
    __C.MODEL.START_TRAIN_CONV_NODE_NAME = __C.MODEL.FEATURE_NODE_NAME
    __C.MODEL.POOL_NODE_NAME = "z.x._.x._.x" # "pool3"
    __C.MODEL.LAST_HIDDEN_NODE_NAME = "z.x" # "h2_d"
__C.MODEL.FEATURE_STRIDE = 16
__C.MODEL.RPN_NUM_CHANNELS = 256
__C.MODEL.ROI_DIM = 6
__C.MODEL.E2E_LR_FACTOR = 1.0
__C.MODEL.RPN_LR_FACTOR = 1.0
__C.MODEL.FRCN_LR_FACTOR = 1.0
