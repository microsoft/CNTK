# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from operator import mul
from functools import reduce

import numpy as np

import cntk
import cntk.io.transforms as xforms
from cntk import ops, io, learners, internal
from cntk import Trainer
from cntk.layers.blocks import BlockFunction


class BlockApiSetup(object):
    '''
     Implement some special requirement ops
    '''
    @staticmethod
    def convolution(output, kernel, stride, pad, kernel_init, bias_init, group, dilation, name):
        '''
         Implement convolution ops

        Args:
            output (int): the output channel size
            kernel (list): the kernel size of filter, with format [width, height]
            stride (list): the stride of convolution, with format [w_stride, h_stride]
            pad (bool): auto padding or not
            kernel_init (`np.array`): the tensor saving initialize values of filter
            bias_init (`np.array`): the tensor saving initialize values of bias
            group (int): the group size in the convolution
            dilation (list): the dilation of convolution, with format [w_dilation, h_dilation]
            name (str): the name of ops

        Return:
            :func:`~cntk.ops.as_block`: the function contains convolution ops
        '''
        def _conv_ops(weights, data):
            return ops.convolution(weights, data, strides=(cntk.InferredDimension, ) + \
                                   ops.sanitize_shape(stride), auto_padding=[False, pad, pad])

        def _weights_parameter(output_channels, init, group_name):
            dilation_kernel = [(k - 1) * d + 1 for k, d in zip(kernel, dilation)]
            # expand kernel to simulate dilation
            used_init = init.copy()
            if dilation_kernel != kernel:
                for axis in range(len(dilation)):
                    kernel_sequence = [x * dilation[axis] for x in range(kernel[axis])]
                    insert_lines = list(set([x for x in range(dilation_kernel[axis])]) ^
                                        set(kernel_sequence))
                    for index in range(len(insert_lines)):
                        insert_lines[index] -= index
                    used_init = np.insert(used_init, insert_lines, 0, axis=len(init.shape) - axis - 1)
            return ops.parameter(shape=(output_channels, cntk.InferredDimension) + ops.sanitize_shape(dilation_kernel),
                                 init=used_init, name=group_name)

        if group == 1:
            w = _weights_parameter(output, kernel_init, '.'.join((name, 'W')))
        else:
            sub_output_channels = int(output / group)
            groups_kernel_init = np.split(kernel_init, group)
            groups_kernel = [_weights_parameter(sub_output_channels, groups_kernel_init[i], 
                             '.'.join((name, str(i), 'W'))) for i in range(0, group)]
            sub_input_channels = groups_kernel[0].shape[1]
        if bias_init is not None:
            b = ops.parameter(shape=(output, ), init=bias_init, name='.'.join((name, 'b')))

        @BlockFunction('Convolution', name)
        def _convolution(x):
            if group == 1:
                apply_x = _conv_ops(w, x)
            else:
                groups_data = [ops.slice(x, axis=0, begin_index=i * sub_input_channels,
                                         end_index=(i + 1) * sub_input_channels) for i in range(0, group)]
                apply_sub = [_conv_ops(group_kernel, group_data)
                             for group_kernel, group_data in zip(groups_kernel, groups_data)]
                apply_x = ops.splice(*apply_sub, axis=0)
            if bias_init is not None:
                apply_x += b
            return apply_x
        return _convolution

    @staticmethod
    def linear(output_shape, input_shape, scale_init, bias_init, name):
        '''
         Implement linear ops, also known as full connection in Caffe
        
        Args:
            output_shape (tuple): the output channel size
            input_shape (tuple): the input channel size
            scale_init (`np.array`): the tensor saving initialize values of scale
            bias_init (`np.array`): the tensor saving initialize values of bias
            name (str): the name of ops

        Return:
            :func:`~cntk.ops.as_block`: the function contains linear ops
        '''
        sc = ops.parameter(shape=input_shape + output_shape, init=scale_init, name='.'.join((name, 'sc')))
        b = ops.parameter(shape=output_shape, init=bias_init, name='.'.join((name, 'b')))

        @BlockFunction('linear', name)
        def _linear(x):
            apply_x = ops.times(x, sc)
            apply_x += b
            return apply_x
        return _linear

    @staticmethod
    def lrn(k, n, alpha, beta, name):
        '''
         Implement LRN ops

        Args:
            k (int): the factor k in LRN
            n (int): the normalization radius
            alpha (float): alpha factor in LRN
            beta (float): beta factor in LRN
            name (str): the name of ops

        Return:
            :func:`~cntk.ops.as_block`: the function contains lrn ops
        '''
        # @BlockFunction('lrn', name)
        # def _lrn(x):
        #     x2 = cntk.ops.square(x)
        #     x2s = cntk.ops.reshape(x2, (1, cntk.InferredDimension), 0, 1)
        #     w = cntk.ops.constant(alpha / (2 * n - 1), (1, 2 * n - 1, 1, 1), name='W')
        #     y = cntk.ops.convolution(w, x2s)
        #     # reshape back to remove the fake singleton reduction dimension
        #     b = cntk.ops.reshape(y, cntk.InferredDimension, 0, 2)
        #     den = cntk.ops.exp(beta * cntk.ops.log(k + b))
        #     apply_x = cntk.ops.element_divide(x, den)
        #     return apply_x
        @BlockFunction('lrn', name)
        def _lrn(x):
            return cntk.local_response_normalization(x, int(n - 1), k, alpha, beta)
        return _lrn


class ApiSetup(object):
    '''
     Setup CNTK ops with given parameters
    '''
    @staticmethod
    def convolution(cntk_layer, inputs):
        '''
         Setup convolution op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of convolution op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk convolution op
        '''
        sanitize_input = internal.sanitize_input(inputs[0])
        params = cntk_layer.parameters
        output_channel = params.output
        kernel_size = params.kernel
        kernel_shape = (output_channel, int(sanitize_input.shape[0] / params.group)) + tuple(kernel_size)
        kernel_init = None
        if cntk_layer.parameter_tensor:
            kernel_data_tensor = cntk_layer.parameter_tensor[0]
            kernel_init = np.asarray(kernel_data_tensor.data, dtype=np.float32)
            kernel_init = np.reshape(kernel_init, newshape=kernel_shape)
        bias_shape = (output_channel, ) + (1,) * 2
        bias_init = None
        if params.need_bias:
            if cntk_layer.parameter_tensor:
                bias_data_tensor = cntk_layer.parameter_tensor[1]
                bias_init = np.asarray(bias_data_tensor.data, dtype=np.float32)
                bias_init = np.reshape(bias_init, bias_shape)
        return BlockApiSetup.convolution(output_channel, kernel_size, stride=params.stride, pad=params.auto_pad,
                                         kernel_init=kernel_init, bias_init=bias_init,
                                         group=params.group, dilation=params.dilation,
                                         name=cntk_layer.op_name)(sanitize_input)

    @staticmethod
    def batch_norm(cntk_layer, inputs):
        '''
         Setup batch normalization op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of batch normalization op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk batch normalization op
        '''
        sanitize_input = internal.sanitize_input(inputs[0])
        parameter_tensor = (sanitize_input.shape[0], )
        scale_init = 1
        bias_init = 0
        mean_init = 1
        var_init = 0
        if cntk_layer.parameter_tensor:
            if len(cntk_layer.parameter_tensor) < 3:
                raise AssertionError('At least three tensors (saved_mean, saved_variance and scale) are needed')
            mean_tensor = cntk_layer.parameter_tensor[0]
            variance_tensor = cntk_layer.parameter_tensor[1]
            global_scale = cntk_layer.parameter_tensor[2].data[0]
            moving_average_factor = 1 / global_scale if global_scale != 0 else 0
            mean_init = np.asarray(mean_tensor.data, dtype=np.float32) * moving_average_factor
            var_init = np.asarray(variance_tensor.data, dtype=np.float32) * moving_average_factor
            if len(cntk_layer.parameter_tensor) == 5:
                scale_tensor = cntk_layer.parameter_tensor[3]
                bias_tensor = cntk_layer.parameter_tensor[4]
                scale_init = np.asarray(scale_tensor.data, dtype=np.float32)
                bias_init = np.asarray(bias_tensor.data, dtype=np.float32)

        scale_parameters = ops.parameter(parameter_tensor, init=scale_init, name='.'.join((cntk_layer.op_name, 'scale')))
        bias_parameters = ops.parameter(parameter_tensor, init=bias_init, name='.'.join((cntk_layer.op_name, 'bias')))
        mean_parameters = ops.parameter(parameter_tensor, init=mean_init, name='.'.join((cntk_layer.op_name, 'mean')))
        var_parameters = ops.parameter(parameter_tensor, init=var_init, name='.'.join((cntk_layer.op_name, 'var')))
        epsilon = cntk_layer.parameters.epsilon

        return ops.batch_normalization(sanitize_input, scale_parameters, bias_parameters, mean_parameters,
                                       var_parameters, True, use_cudnn_engine=False, epsilon=epsilon,
                                       running_count=ops.constant(0),
                                       name=cntk_layer.op_name)

    @staticmethod
    def pooling(cntk_layer, inputs):
        '''
         Setup pooling op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of pooling op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk pooling op
        '''
        sanitize_input = internal.sanitize_input(inputs[0])
        pooling_type = ops.PoolingType_Average if cntk_layer.parameters.pooling_type else ops.PoolingType_Max
        return ops.pooling(sanitize_input, pooling_type, tuple(cntk_layer.parameters.kernel),
                           strides=tuple(cntk_layer.parameters.stride),
                           auto_padding=[cntk_layer.parameters.auto_pad],
                           ceil_out_dim=True,
                           name=cntk_layer.op_name)

    @staticmethod
    def relu(cntk_layer, inputs):
        '''
         Setup ReLU op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of ReLU op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk ReLU op
        '''
        sanitize_input = internal.sanitize_input(inputs[0])
        return ops.relu(sanitize_input, name=cntk_layer.op_name)

    @staticmethod
    def dense(cntk_layer, inputs):
        '''
         Setup dense op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of dense op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk dense op
        '''
        sanitize_input = internal.sanitize_input(inputs[0])
        input_channel = sanitize_input.shape
        output_channel = cntk_layer.parameters.num_output

        flattened_channel = reduce(mul, list(input_channel))
        scale_shape = input_channel + (output_channel, )
        bias_shape = (output_channel, )

        if cntk_layer.parameter_tensor:
            if len(cntk_layer.parameter_tensor) != 2:
                raise AssertionError('dense layer layer receives two inputs (scale/bias)')
            scale_tensor = cntk_layer.parameter_tensor[0]
            bias_tensor = cntk_layer.parameter_tensor[1]
            scale_init = np.asarray(scale_tensor.data, np.float32)
            if cntk_layer.parameters.transpose:
                scale_init = np.reshape(scale_init, (output_channel, flattened_channel))
                scale_init = np.transpose(scale_init).copy()
                scale_init = np.reshape(scale_init, scale_shape)
            else:
                scale_init = np.reshape(scale_init, scale_shape)
            bias_init = np.asarray(bias_tensor.data, np.float32)
        return BlockApiSetup.linear(bias_shape, scale_shape, scale_init, bias_init, cntk_layer.op_name)(sanitize_input)

    @staticmethod
    def plus(cntk_layer, inputs):
        '''
         Setup plus op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of dense op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk dense op
        '''
        sanitize_left = ops.sanitize_input(inputs[0])
        sanitize_right = ops.sanitize_input(inputs[1])
        return ops.plus(sanitize_left, sanitize_right, name=cntk_layer.op_name)

    @staticmethod
    def dropout(cntk_layer, inputs):
        '''
         Setup dropout op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of dropout op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or 
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk dropout op
        '''
        sanitize_output = ops.sanitize_input(inputs[0])
        return ops.dropout(sanitize_output, name=cntk_layer.op_name)

    @staticmethod
    def lrn(cntk_layer, inputs):
        '''
         Setup lrn op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of lrn op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk lrn op
        '''
        sanitize_output = ops.sanitize_input(inputs[0])
        params = cntk_layer.parameters
        return BlockApiSetup.lrn(params.k, params.kernel_size, params.alpha,
                                 params.beta, cntk_layer.op_name)(sanitize_output)

    @staticmethod
    def splice(cntk_layer, inputs):
        '''
         Setup splice op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of splice op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk splice op
        '''
        return ops.splice(*inputs, axis=0, name=cntk_layer.op_name)

    @staticmethod
    def softmax(cntk_layer, inputs):
        '''
         Setup softmax op with given parameters

        Args:
            cntk_layer (:class:`~cntk.contrib.crosstalkcaffe.unimodel.cntkmodel.CntkLayersDefinition`):
                the layer definition of softmax op
            inputs (list): a list contains all :class:`~cntk.ops.functions.Function` or
                :class:`~cntk.input`

        Return:
            :func:`~cntk.ops.functions.Function`: instaced cntk softmax op
        '''
        sanitize_output = ops.sanitize_input(inputs[0])
        return ops.softmax(sanitize_output, name=cntk_layer.op_name)


class CntkApiInstance(object):
    '''
     Instace CNTK ops and network
    '''
    def __init__(self, cntk_uni_model, global_conf):
        self._functions = {}
        self._output = None
        self._model_solver = global_conf.model_solver
        self._source_solver = global_conf.source_solver
        self._instance(cntk_uni_model)

    def _instance(self, cntk_uni_model):
        self.instance_input(cntk_uni_model.data_provider)
        self.instance_functions(cntk_uni_model.cntk_sorted_layers, cntk_uni_model.cntk_layers)

    def instance_input(self, data_providers):
        '''
         Instace the inputs into CNTK variable

        Args:
            data_providers (list): the list contains the definition of inputs

        Return:
            None
        '''
        if self._model_solver.cntk_tensor is not None:
            for key, tensor in self._model_solver.cntk_tensor.items():
                input_var = cntk.input(tuple(tensor), name=key)
                self._functions[key] = input_var
        else:
            for data_provider in data_providers:
                input_var = cntk.input(tuple(data_provider.tensor[:]), name=data_provider.op_name)
                self._functions[data_provider.op_name] = input_var

    def instance_functions(self, cntk_sorted_layers, cntk_layers):
        '''
         Instace all nodes into CNTK ops

        Args:
            cntk_sorted_layers (list): the list contains the name of instaced layers with 
                traversal order
            cntk_layers (dict): the dict contains all layers definition

        Return:
            None
        '''
        unused_func = set()
        for cntk_sorted_layer in cntk_sorted_layers:
            cntk_layer = cntk_layers[cntk_sorted_layer]
            local_inputs = []
            for local_input in cntk_layer.inputs:
                local_inputs.append(self._functions[local_input])
                if self._functions[local_input] in unused_func:
                    unused_func.remove(self._functions[local_input])
            self._functions[cntk_layer.op_name] = getattr(ApiSetup, cntk_layer.op_type.name\
                                                          )(cntk_layer, local_inputs)
            unused_func.add(self._functions[cntk_layer.op_name])
        self._output = ops.combine(list(unused_func), name='outputs')

    def export_model(self):
        '''
         Save instanced CNTK model

        Args:
            None

        Return:
            None
        '''
        save_path = self._model_solver.cntk_model_path
        self._output.save(save_path)

    def get_model(self):
        '''
         Get instaced CNTK model

        Args:
            None

        Return:
            :func:`~cntk.ops.functions.Function`: the output node of CNTK
        '''
        return self._output

    def get_functions(self):
        '''
         Return the functions of CNTK network

        Args:
            None
        
        Return:
            list: the instaced functions of CNTK
        '''
        return self._functions
