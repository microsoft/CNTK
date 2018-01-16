# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from enum import Enum


class CntkParameters(object):
    '''
     The parameter of all CNTK op
    '''
    def __init__(self):
        pass


class CntkConvolutionParameters(CntkParameters):
    '''
     The parameter definition of convolution op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.output = 0
        self.stride = [0, 0]
        self.kernel = [0, 0]
        self.auto_pad = False
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]
        self.need_bias = True
        self.dilation = [1, 1]
        self.group = 1


class CntkPoolingParameters(CntkParameters):
    '''
     The parameter definition of pooling op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.stride = [0, 0]
        self.kernel = [0, 0]
        self.auto_pad = False
        self.pooling_type = 0    # 0 for max, 1 for average


class CntkBatchNormParameters(CntkParameters):
    '''
     The parameter definition of batch normalization op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.spatial = 2
        self.norm_time_const = 0
        self.blend_time_const = 0
        self.epsilon = 0.00001
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]


class CntkDenseParameters(CntkParameters):
    '''
     The parameter definition of dense op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.num_output = 0
        self.scale_setting = [1, 1]
        self.bias_setting = [1, 1]
        self.transpose = False


class CntkSpliceParameters(CntkParameters):
    '''
     The parameter definition of splice op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.axis = 1


class CntkLRNParameters(CntkParameters):
    '''
     The parameter definition of LRN op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.kernel_size = 5
        self.alpha = 1
        self.beta = 5
        self.k = 1


class CntkPSROIPoolingParameters(CntkParameters):
    '''
     The parameter definition of PSROIPooling op
    '''
    def __init__(self):
        CntkParameters.__init__(self)
        self.group_size = 1
        self.out_channel = 1


class CntkLayerType(Enum):
    '''
     The enumate of CNTK ops
    '''
    relu = 1
    convolution = 2
    pooling = 3
    batch_norm = 4
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
    '''
     The definition of data blob
    '''
    def __init__(self):
        self.tensor = [0 * 4]
        self.data = []


class CntkLayersDefinition(object):
    '''
     The definition of nodes, created by Caffe and instaced by CNTK
    '''
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.op_name = None
        self.parameters = None
        self.op_type = CntkLayerType.unknown
        self.tensor = []
        self.parameter_tensor = []


class CntkSolver(object):
    '''
     Record the solver state
    '''
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
    '''
     Record the basic information of model
    '''
    def __init__(self):
        self.data_provider = []
        self.cntk_layers = {}              # Dict Key: function name, Value: function definition
        self.cntk_sorted_layers = []       # List Key: function name
        self.model_name = 'Untitled'
        self.solver = None

