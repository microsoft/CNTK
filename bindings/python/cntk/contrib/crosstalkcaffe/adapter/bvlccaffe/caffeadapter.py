# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
import time
import math

from cntk.contrib.crosstalkcaffe.unimodel import cntkmodel
from cntk.contrib.crosstalkcaffe.adapter import baseadapter
from . import caffeimpl

try:
    from google.protobuf import text_format
except ImportError:
    sys.stderr.write('Parsing the weights needs Protobuf\n')

def _format_to_list(target, rank, default_pad=0):
    if isinstance(target, int):
        list_target = [target] * rank
    else:
        list_target = target
        if len(target) != rank:
            list_target = [default_pad if not len(target) else target[0]] * rank
    return list_target


def _setup_convolution_parameters(parameters, input_tensor, is_ceil_pad=False):
    # considering the ceil and floor padding of bvlccaffe
    kernel_size = _format_to_list(parameters.kernel_size, 2)
    kernel_size[0] = parameters.kernel_h or kernel_size[0]
    kernel_size[1] = parameters.kernel_w or kernel_size[1]
    strides = _format_to_list(parameters.stride, 2, 1)
    strides[0] = parameters.stride_h or strides[0]
    strides[1] = parameters.stride_w or strides[1]
    lower_pad = _format_to_list(parameters.pad, 2, 0)
    lower_pad[0] = parameters.pad_h or lower_pad[0]
    lower_pad[1] = parameters.pad_w or lower_pad[1]
    dilation = [1] * 2
    if hasattr(parameters, 'dilation') and len(parameters.dilation) != 0:
        dilation[0] = parameters.dilation[0]
        dilation[1] = parameters.dilation[1] if len(parameters.dilation) > 1 else parameters.dilation[0]
    input_size = input_tensor[1:]
    output_size = [0] * 2
    for i in range(2):
        kernel_expand = (kernel_size[i] - 1) * dilation[i] + 1
        precise_size = float(input_size[i] + 2 * lower_pad[i] - kernel_expand) / strides[i] + 1
        output_size[i] = int(math.ceil(precise_size)) if is_ceil_pad else int(math.floor(precise_size))
    pad = True if lower_pad[0] or lower_pad[1] else False
    return kernel_size, strides, output_size, pad, dilation


class SetupCaffeParameters(object):
    '''
     Setup Caffe parameters into CNTK format
    '''
    @staticmethod
    def default(caffe_parameters, inputs_info, cntk_layer_def, tensor_check=True):
        '''
         The default Caffe to CNTK uniform model setup

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers
            tensor_check (bool, default=True): Whether to check the tensor shape

        Return:
            None
        '''
        if caffe_parameters:
            pass
        # tensor align check
        if not tensor_check:
            cntk_layer_def.parameters = cntkmodel.CntkParameters()
            return
        try:
            identity_check = inputs_info[0].tensor[:]
            for input_info in inputs_info:
                if not input_info.tensor == identity_check:
                    raise AttributeError('Non-align input tensor %s', cntk_layer_def.op_name)
        except OverflowError:
            raise AttributeError('Non-align input tensor %s', cntk_layer_def.op_name)

        cntk_layer_def.tensor = identity_check
        if not cntk_layer_def.parameters:
            cntk_layer_def.parameters = cntkmodel.CntkParameters()

    @staticmethod
    def convolution(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The convolution parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkConvolutionParameters()

        kernel_size, strides, output_size, pad, dilation = \
            _setup_convolution_parameters(caffe_parameters, inputs_info[0].tensor)

        # load the output channel numbers and bias setting
        cntk_layer_def.parameters.output = caffe_parameters.num_output
        cntk_layer_def.parameters.need_bias = caffe_parameters.bias_term
        cntk_layer_def.parameters.group = caffe_parameters.group

        cntk_layer_def.parameters.kernel = kernel_size
        cntk_layer_def.parameters.stride = strides
        cntk_layer_def.parameters.auto_pad = pad
        cntk_layer_def.parameters.dilation = dilation
        cntk_layer_def.tensor = [caffe_parameters.num_output, output_size[0], output_size[1]]

    @staticmethod
    def pooling(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The pooling parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkPoolingParameters()

        # To support global pooling
        if caffe_parameters.global_pooling:
            filter_shape = (inputs_info[0].tensor[1], inputs_info[0].tensor[2])
            strides = (1, 1)
            output_size = (1, 1)
            pad = False
        else:
            filter_shape, strides, output_size, pad, _ = _setup_convolution_parameters(
                caffe_parameters, inputs_info[0].tensor, is_ceil_pad=True)

        cntk_layer_def.parameters.kernel = filter_shape
        cntk_layer_def.parameters.stride = strides
        cntk_layer_def.parameters.auto_pad = pad
        cntk_layer_def.parameters.pooling_type = caffe_parameters.pool
        cntk_layer_def.tensor = [inputs_info[0].tensor[0], output_size[0], output_size[1]]

    @staticmethod
    def batch_norm(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The batch normalization parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkBatchNormParameters()
        cntk_layer_def.parameters.epsilon = caffe_parameters.eps
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

    @staticmethod
    def dense(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The dense parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        if inputs_info:
            pass
        cntk_layer_def.parameters = cntkmodel.CntkDenseParameters()

        cntk_layer_def.parameters.transpose = True if not caffe_parameters.transpose else False
        cntk_layer_def.parameters.num_output = caffe_parameters.num_output
        cntk_layer_def.tensor = [caffe_parameters.num_output]

    @staticmethod
    def splice(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The splice parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkSpliceParameters()

        if caffe_parameters is not None:
            cntk_layer_def.parameters.axis = caffe_parameters.axis
        else:
            cntk_layer_def.parameters.axis = 1
        output_tensor = inputs_info[0].tensor[:]
        output_tensor[0] = 0
        for input_info in inputs_info:
            if not output_tensor[1:] == input_info.tensor[1:]:
                raise IndexError('Non-align tensor information\n')
            output_tensor[0] += input_info.tensor[cntk_layer_def.parameters.axis - 1]
        cntk_layer_def.tensor = output_tensor

    @staticmethod
    def relu(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The ReLU parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

    @staticmethod
    def plus(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The plus parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

    @staticmethod
    def dropout(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The dropout parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

    @staticmethod
    def lrn(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The lrn parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkLRNParameters()
        cntk_layer_def.parameters.kernel_size = (caffe_parameters.local_size + 1) / 2
        cntk_layer_def.parameters.alpha = caffe_parameters.alpha
        cntk_layer_def.parameters.beta = caffe_parameters.beta
        cntk_layer_def.parameters.k = caffe_parameters.k
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

    @staticmethod
    def psroi_pooling(caffe_parameters, _, cntk_layer_def):
        '''
         The psroipooling parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        cntk_layer_def.parameters = cntkmodel.CntkPSROIPoolingParameters()
        cntk_layer_def.parameters.group_size = caffe_parameters.group_size
        cntk_layer_def.parameters.out_channel = caffe_parameters.output_dim
        cntk_layer_def.tensor = [caffe_parameters.output_dim, caffe_parameters.group_size, caffe_parameters.group_size]

    @staticmethod
    def softmax(caffe_parameters, inputs_info, cntk_layer_def):
        '''
         The softmax parameter setup from Caffe to CNTK

        Args:
            caffe_parameters (:class:`caffe.Parameters`): the parameters of Caffe
            inputs_info ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkTensorDefinition`):
                The input information of current layer
            cntk_layer_def ('class':`cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                The converted definition of CNTK layers

        Return:
            None
        '''
        SetupCaffeParameters.default(caffe_parameters, inputs_info, cntk_layer_def)

NEGLECT_LAYERS = ['Scale', 'Dropout']


class CaffeAdapter(baseadapter.Adapter):
    '''
     Adapt Caffe prototxt and weights into uniform format
    '''
    def __init__(self):
        baseadapter.Adapter.__init__(self)
        self._raw_net = None
        self._raw_solver = None
        self._uni_model = None
        self._source_solver = None
        self._caffe_pb2 = None
        return

    def load_model(self, global_conf):
        '''
         Load prototxt and weights of Caffe, parsing them into uniform model

        Args:
            global_conf (:class:`~cntk.contrib.crosstalkcaffe.utils.globalconf.GlobalConf`):
                The global configurations
        
        Return:
            None
        '''
        sys.stdout.write('Start loading model:\n')
        self._source_solver = global_conf.source_solver

        # loading the model
        if not os.path.exists(self._source_solver.model_path):
            sys.stderr.write('Check the file name and path of input model path\n')
            sys.exit()
        caffe_impl = caffeimpl.CaffeResolver()
        self._uni_model = cntkmodel.CntkModelDescription()
        self._raw_solver = caffe_impl.solver()
        self._raw_net = caffe_impl.net()
        self._caffe_pb2 = caffe_impl.caffepb
        with open(self._source_solver.model_path, 'r') as net_file:
            text_format.Merge(net_file.read(), self._raw_net)        
        self._uni_model.model_name = os.path.splitext(self._source_solver.model_path)[0]

        self._adapt_data_provider()
        self._adapt_net()

        # loading the weights
        if not os.path.exists(self._source_solver.weights_path):
            sys.stderr.write('Check the file name and path of weights file\n')
            sys.exit()
        else:
            self._adapt_parameter_data()

        sys.stdout.write('Finished model loading\n')

        return self._uni_model

    def adapt_solver(self):
        '''
         Adapt Caffe solver into CNTK format
        
        Args:
            None
        
        Return:
            None
        '''
        self._uni_model.solver = cntkmodel.CntkSolver()
        solver = self._uni_model.solver
        caffe_solver = self._raw_solver

        # Get the casting between iterations and epochs
        if 'iters_per_epoch' in self._source_solver.keys():
            iter_per_epoch = self._source_solver['iters_per_epoch']
            solver.adjust_interval = int(caffe_solver.stepsize / iter_per_epoch + 0.5)
            solver.max_epoch = int(caffe_solver.max_iter / iter_per_epoch + 0.5)

        solver.learning_rate = caffe_solver.base_lr
        solver.decrease_factor = caffe_solver.gamma
        solver.momentum = caffe_solver.momentum
        solver.weight_decay = caffe_solver.weight_decay
        solver.number_to_show_result = caffe_solver.display

    def _adapt_data_provider(self):
        if not self._raw_net:
            raise KeyError('Invalid net structure\n')
        raw_layers = self._raw_net.layer or self._raw_net.layers
        for raw_layer in raw_layers:
            if raw_layer.type != 'Input':
                continue
            if not self._inclusive_layer(raw_layer):
                continue

            for i in range(0, len(raw_layer.top)):
                data_provider = cntkmodel.CntkLayersDefinition()
                data_provider.op_name = raw_layer.top[i]
                data_provider.tensor = raw_layer.input_param.shape[i].dim[1:] # cancel mini-batch
                self._uni_model.data_provider.append(data_provider)

    def _adapt_net(self):
        raw_layers = self._raw_net.layer or self._raw_net.layers
        uni_model = self._uni_model

        data_providers = self._uni_model.data_provider
        scope_inputs = {}
        for data_provider in data_providers:
            scope_inputs[data_provider.op_name] = data_provider

        for raw_layer in raw_layers:
            if raw_layer.type == 'Input':
                continue
            cntk_layer_type, caffe_layer_type = self._get_layer_type(raw_layer)
            if cntk_layer_type is None:
                if raw_layer.type not in NEGLECT_LAYERS:
                    sys.stderr.write('Non-decision type of CNTK\n')
                    raise AssertionError('Dangerous call\n')
                if raw_layer.type == 'Scale' and scope_inputs[raw_layer.bottom[0]].op_type\
                        == cntkmodel.CntkLayerType.batch_norm:
                    pass
                else:
                    sys.stderr.write('dangerous call of %s', raw_layer.name)
                bottom_name = raw_layer.bottom[0]
                top_name = raw_layer.top[0]
                scope_inputs[top_name] = scope_inputs[bottom_name]
                continue

            # inputs
            bottom_names = raw_layer.bottom
            inputs_info = []
            for bottom_name in bottom_names:
                inputs_info.append(scope_inputs[bottom_name])

            # function
            cntk_layer_def = self._setup_cntk_layer_def(cntk_layer_type, raw_layer, inputs_info)

            # refresh the lists
            # only support single output operation
            if len(raw_layer.top) > 1:
                sys.stderr.write('Single output layers allowed currently: %s.%s\n' \
                                 % (str(cntk_layer_type), raw_layer.name))
            top_names = raw_layer.top[0]
            scope_inputs[top_names] = cntk_layer_def

            # push into vectors
            uni_model.cntk_layers[raw_layer.name] = cntk_layer_def
            uni_model.cntk_sorted_layers.append(raw_layer.name)

    def _adapt_parameter_data(self):
        sys.stdout.write('start parameter loading...\n')
        start_time = time.time()
        caffe_impl = caffeimpl.CaffeResolver()
        if caffe_impl.runtime():
            paras = caffe_impl.caffe.Net(self._source_solver.model_path,
                                         self._source_solver.weights_path, caffe_impl.caffe.TEST).params
            caffe_blobs = [(layer_name, map(lambda blobs: blobs.data, blob_vec))
                           for layer_name, blob_vec in paras.items()]
        else:
            sys.stdout.write('loading weights via protopb, may take longer time than runtime')
            params = caffe_impl.net()
            params.MergeFromString(open(self._source_solver.weights_path, 'rb').read())
            params_layers = params.layer or params.layers
            caffe_blobs = [(layer.name, [blob.data for blob in layer.blobs]) for layer in params_layers if layer.blobs]
        sys.stdout.write('finished loading, total time: %d\n' % (time.time() - start_time))

        # mapping the script into layers
        sys.stdout.write('start parameter matching...\n')
        for caffe_blob in caffe_blobs:
            try:
                cntk_layer = self._uni_model.cntk_layers[caffe_blob[0]]
            except KeyError:
                caffe_layers = self._raw_net.layer or self._raw_net.layers
                try:
                    special_layer = [layer for layer in caffe_layers if layer.name == caffe_blob[0]][0]
                    if special_layer.type == 'Scale':
                        previous_layer = [caffe_layers[i - 1] for i in range(len(caffe_layers))
                                          if caffe_layers[i] == special_layer][0]
                        if previous_layer.type != 'BatchNorm':
                            raise AssertionError('un-support pure Scale layer without BN in %s' % caffe_layers.name)
                        cntk_layer = self._uni_model.cntk_layers[previous_layer.name]
                    else:
                        raise AssertionError('un-match layer name %s while matching parameters\n' % caffe_blob.name)
                except IndexError:
                    sys.stderr.write('ignore weights for %s, since not contained in graph\n' % caffe_blob[0])
            for blob in caffe_blob[1]:
                cntk_parameter_tensor = cntkmodel.CntkTensorDefinition()
                cntk_parameter_tensor.data = blob
                cntk_layer.parameter_tensor.append(cntk_parameter_tensor)
        sys.stdout.write('finished matching.\n')

    def _get_layer_type(self, raw_layer):
        caffe_layer_type = raw_layer.type
        try:
            cntk_layer_type = caffeimpl.CAFFE_LAYER_WRAPPER[caffe_layer_type]
        except KeyError:
            cntk_layer_type = self._try_special_case_wrapper(raw_layer)
        if not cntk_layer_type:
            if raw_layer.type in NEGLECT_LAYERS:
                sys.stderr.write('Warning: Un-supported bvlccaffe type: %s-%s\n' % (raw_layer.name, caffe_layer_type))
            else:
                sys.stderr.write('Error: Un-support and import bvlccaffe type missing: %s-%s\n'
                                 % (raw_layer.name, caffe_layer_type))
                raise KeyError('.'.join((raw_layer.name, caffe_layer_type)))
            return None, None
        return cntk_layer_type, caffe_layer_type

    @staticmethod
    def _get_layer_parameters(raw_layer):
        convert_name = raw_layer.type.lower() + 'param'
        for (descriptor, attr) in raw_layer.ListFields():
            if descriptor.name.lower().replace('_','') == convert_name:
                return attr
        return None

    def _try_special_case_wrapper(self, raw_layer):
        layer_parameter = self._get_layer_parameters(raw_layer)
        cntk_layer_type = None
        if raw_layer.type == 'Eltwise':
            operate_name = self._caffe_pb2.EltwiseParameter.EltwiseOp.DESCRIPTOR.values_by_number[layer_parameter.operation].name
            layer_type = '_'.join((raw_layer.type, operate_name))
            cntk_layer_type = caffeimpl.CAFFE_LAYER_WRAPPER[layer_type]
        return cntk_layer_type

    def _setup_cntk_layer_def(self, cntk_layer_type, raw_layer, inputs_info):
        cntk_layer_def = cntkmodel.CntkLayersDefinition()
        cntk_layer_def.op_name = raw_layer.name
        cntk_layer_def.op_type = cntk_layer_type
        for input_info in inputs_info:
            cntk_layer_def.inputs.append(input_info.op_name)
        cntk_layer_def.outputs.append(raw_layer.name)
        getattr(SetupCaffeParameters, cntk_layer_type.name)(self._get_layer_parameters(raw_layer),
                                                            inputs_info, cntk_layer_def)
        return cntk_layer_def

    def _inclusive_layer(self, raw_layer):
        phase = self._source_solver.phase
        if len(raw_layer.include):
            phase = raw_layer.include[0].phase
        if len(raw_layer.exclude):
            phase = 1 - raw_layer.exclude[0].phase
        return 1 if phase == self._source_solver.phase else 0
