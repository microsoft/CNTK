# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from enum import Enum


class CntkParameters(object):
    def __init__(self): pass


class CntkConvolutionParameters(CntkParameters):
    def __init__(self):
        self.output = 0
        self.stride = [0, 0]
        self.kernel = [0, 0]
        self.auto_pad = False
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]
        self.need_bias = True
        self.dilation = 1
        self.group = 1


class CntkPoolingParameters(CntkParameters):
    def __init__(self):
        self.stride = [0, 0]
        self.kernel = [0, 0]
        self.auto_pad = False
        self.pooling_type = 0    # 0 for max, 1 for average


class CntkBatchNormParameters(CntkParameters):
    def __init__(self):
        self.spatial = 2
        self.norm_time_const = 0
        self.blend_time_const = 0
        self.epsilon = 0.00001
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]


class CntkDenseLayerParameters(CntkParameters):
    def __init__(self):
        self.num_output = 0
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]
        self.transpose = False


class CntkSpliceParameters(CntkParameters):
    def __init__(self):
        self.axis = 1


class CntkClassificationParameters(CntkParameters):
    def __init__(self):
        self.top_n = 1


class CntkLRNParameters(CntkParameters):
    def __init__(self):
        self.kernel_size = 5
        self.alpha = 1
        self.beta = 5
        self.k = 1


class CntkPSROIPoolingParameters(CntkParameters):
    def __init__(self):
        self.group_size = 1
        self.out_channel = 1


class CntkLayerType(Enum):
    relu = 1
    convolution = 2
    pooling = 3
    batch_normalization = 4
    plus = 5
    dense = 6
    splice = 7
    classification_error = 10
    cross_entropy_with_softmax = 11
    dropout = 12
    lrn = 13
    psroi_pooling = 14
    softmax = 15
    unknown = 100


class CntkTensorDefinition(object):
    def __init__(self):
        self.tensor = [0 * 4]
        self.data = []


# Currently, the function and layer of CNTK have some
class CntkLayersDefinition(object):
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.op_name = None
        self.parameters = None
        self.op_type = CntkLayerType.unknown
        self.tensor = []
        self.parameter_tensor = []


class CntkSolver(object):
    def __init__(self):
        self.learning_rate = None
        self.max_epoch = None
        self.adjust_interval = None
        self.decrease_factor = None
        self.upper_limit = None
        self.minibatch_size = None
        self.weight_decay = None
        self.dropout = None
        self.number_to_show_result = None
        self.grad_update_type = None
        self.momentum = None


class CntkModelDescription(object):
    def __init__(self):
        self.data_provider = []
        self.cntk_layers = {}              # Dict Key: function name, Value: function definition
        self.cntk_sorted_layers = []       # List Key: function name
        self.model_name = 'Untitled'
        self.solver = None

