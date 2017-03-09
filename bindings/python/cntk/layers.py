# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# layers -- blocks in the network that are used layer-like, i.e. layered on top of each other
#           e.g. a fully connected layer with non-linearity

# TODO: clean up the dependencies
from __future__ import division
import numpy as np
from .ops import parameter, input_variable, placeholder_variable, combine
from .ops import times, convolution, pooling, batch_normalization, dropout, unpooling
from .utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from .utils import Record, _as_tuple
from .blocks import * # layers.py imports all of blocks and models
from .models import *
from .blocks import _trace_layers  # (debugging)

from .ops.functions import Function
from .ops.variables import Variable

# this is what we initialize weight matrices from by default
from .blocks import _get_current_default_options, _is_given, _initializer_for, _resolve_activation, _INFERRED

# Dense -- create a fully-connected linear projection layer with optional non-linear activation
# Note: shape may describe a tensor as well.
# input_rank given: number of inferred axes to add to W (map_rank must not be given)
# map_rank   given: expand W to leave exactly mapRank axes (input_rank must not be given)
# none       given: expand W to all (same as map_rank=0)
def Dense(shape, init=init_default_or_glorot_uniform, activation=activation_default_or_None,
          input_rank=None, map_rank=None,
          bias=bias_default_or_True, init_bias=init_bias_default_or_0, 
          name=''):
    activation = _resolve_activation(activation)
    bias       = bias if _is_given(bias) else _get_current_default_options().bias
    output_shape = _as_tuple(shape)

    if input_rank is not None and map_rank is not None:
        raise ValueError("Dense: input_rank and map_rank cannot be specified at the same time.")

    # determine meaning of axes
    # W gets dimension (input_shape + shape)
    # where input_shape is determined as:
    #  - by default, equal to the dimensions of the input passed to Dense()
    #  - if input_rank is given, then the last 'input_rank' dimensions of the input (all others are not reduced over)
    #  - if map_rank is given, then the all but the first 'map_rank' dimensions of the input (those are not reduced over)
    # where input_rank and map_rank are mutuallly exclusive.

    #output_rank = -len(output_shape)   # support outputs with tensor layouts
    # BUGBUG: Should this be a negative number now, since output is the last axis in Python?
    output_rank = len(output_shape)   # support outputs with tensor layouts

    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED * (input_rank if input_rank is not None else 1)

    if input_rank is not None:
        UntestedBranchError("Dense, input_rank option not implemented")
        infer_input_rank_to_map = -1 # means map_rank is not specified; input_rank rules
    elif map_rank is None:
        infer_input_rank_to_map = 0  # neither given: default to 'infer W to use all input dims'
    else:
        UntestedBranchError("Dense, map_rank option not implemented")
        infer_input_rank_to_map = map_rank  # infer W to use all input dims except the first static 'map_rank' ones

    # parameters bound to this Function
    init_weights = _initializer_for(init, Record(output_rank=output_rank))
    W = Parameter(input_shape + output_shape, init=init_weights, name='W')
    b = Parameter(              output_shape, init=init_bias,    name='b') if bias else None

    # expression of this function
    x = Placeholder(name='dense_arg')
    apply_x = times(x, W, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)
    if b:
        apply_x = apply_x + b
    apply_x = apply_x >> activation
    return Block(apply_x, 'Dense', name, Record(W=W, b=b), make_block=True)

# Embedding -- create a linear embedding layer
# To create an embedding from a file, use this:
#  Embedding(weights=np.load('PATH'))
def Embedding(shape=None, init=None, weights=None, name=''):
    if init is not None or weights is not None:
        raise ValueError('Embedding: init and weights options are mutually exclusive')

    # parameters bound to this Function:
    # no weights given: learn the embedding
    if weights is None:
        if shape is None:
            raise ValueError('Embedding: output shape must be specified')
        if init is None:
            init = init_default_or_glorot_uniform
        shape = _as_tuple(shape)
        weight_shape = _INFERRED + shape
        E = Parameter(weight_shape, init=init, name='E')
    # weights given: use them as constant
    else:
        UntestedBranchError("Embedding, from constant")
        import numpy as np
        if not isinstance(weights, array): # TODO: is this the correct test for a numpy array
            UntestedBranchError("Embedding, from constant that is not an array")
            # TODO: can 'weights' be a CNTK object? Then how to do this?
            raise ValueError('Embedding: weights must be a numpy array')
        weight_shape = np.shape(weights)
        if shape is not None: # user may give shape, then it must match
            if len(shape) >= len(weight_shape) or weight_shape[-len(shape):] != shape:
                raise ValueError('Embedding: shape parameter must match weights')
        E = Constant(weights, name='E')

    # expression
    x = Placeholder(name='embedding_arg')
    apply_x = times(x, E)
    return Block(apply_x, 'Embedding', name, Record(E=E), make_block=True)

# Convolution -- create a convolution layer with optional non-linearity
#             ( (sample shape) +  (output shape) +  (reduction shape) + (shifting shape) )
#    in     : ( (sample shape) +                 +  (reduction shape) + (shifting shape) )
#    kernel : (                +  (output shape) +  (reduction shape) + (filte  shape)   )
#    out    : ( (sample shape) +  (output shape) +                    + (shifting shape) )
# TODO: Can we specify atrous convolution? How?
# TODO: sharing = false?
# TODO: conflict of parameter order: filter_shape or num_filters first?
#  - filter_shape first is logical for non-NN applications such as straight image filtering
#  - num_filters first is what Keras does
def Convolution(filter_shape,        # e.g. (3,3)
                num_filters=None,    # e.g. 64 or None (which means 1 channel and don't add a dimension_
                activation=activation_default_or_None,
                init=init_default_or_glorot_uniform,
                pad=pad_default_or_False,
                strides=1,
                sharing=True,     # (must be True currently)
                bias=bias_default_or_True,
                init_bias=init_bias_default_or_0,
                reduction_rank=1, # (must be 1 currently)
                max_temp_mem_size_in_samples=0, 
                name=''):
    #UntestedBranchError("Convolution")
    activation = _resolve_activation(activation)
    pad  = pad  if _is_given(pad ) else _get_current_default_options().pad
    bias = bias if _is_given(bias) else _get_current_default_options().bias
    # TODO: there must be a Python trick to do this as a function call on locals or so
    if reduction_rank != 1:
        NotImplementedError("Convolution: reduction_rank other than 1 currently not supported")
    if not sharing:
        NotImplementedError("Convolution: sharing option currently must be True")
    output_channels_shape = _as_tuple(num_filters)
    output_rank = len(output_channels_shape)
    filter_rank = len(filter_shape)
    kernel_shape = _INFERRED * reduction_rank + filter_shape # kernel := filter plus reductionDims

    # parameters bound to this Function
    #init_kernel = glorot_uniform(filter_rank=-filter_rank, output_rank=1)
    init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
    # BUGBUG: It is very confusing that output_rank is negative, esp. since that means count from the start. Solution: add a flag
    W = Parameter(output_channels_shape + kernel_shape,             init=init_kernel, name='W')                   # (K, C, H, W) aka [ W x H x C x K ]
    b = Parameter(output_channels_shape + (1,) * len(filter_shape), init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]

    # expression
    x = Placeholder(name='convolution_arg')
    # TODO: update the parameter order of convolution() to match the optional ones as in here? (options order matches Keras)
    apply_x = convolution (W, x,
                           strides=_as_tuple(strides),
                           sharing=_as_tuple(sharing),
                           auto_padding=_as_tuple(pad),
                           # TODO: can we rename auto_padding to pad?
                           transpose=False,
                           max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
    if bias:
        apply_x = apply_x + b
    apply_x = apply_x >> activation

    op_name = 'Convolution{}D'.format(len(filter_shape))
    return Block(apply_x, op_name, name, Record(W=W, b=b), make_block=True)

# Convolution1D -- create a 1D convolution layer with optional non-linearity
def Convolution1D(filter_shape,        # a scalar, e.g., 3 
                  num_filters=None,
                  activation=activation_default_or_None,
                  init=init_default_or_glorot_uniform,
                  pad=pad_default_or_False,
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=bias_default_or_True,
                  init_bias=init_bias_default_or_0,
                  name=''):
    if len(filter_shape) != 1: 
         raise ValueError('Convolution1D: filter_shape must be a scalar')
    return Convolution(filter_shape, num_filters, activation, init, pad, strides, sharing, bias, init_bias, name=name)

# Convolution2D -- create a 2D convolution layer with optional non-linearity
def Convolution2D(filter_shape,        # a 2D tuple, e.g., (3,3) 
                  num_filters=None,
                  activation=activation_default_or_None,
                  init=init_default_or_glorot_uniform,
                  pad=pad_default_or_False,
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=bias_default_or_True,
                  init_bias=init_bias_default_or_0,
                  name=''):
    if len(filter_shape) != 2: 
         raise ValueError('Convolution2D: filter_shape must be a 2D tuple, e.g. (3,3)')
    return Convolution(filter_shape, num_filters, activation, init, pad, strides, sharing, bias, init_bias, name=name)

# Convolution3D -- create a 3D convolution layer with optional non-linearity
def Convolution3D(filter_shape,        # a 3D tuple, e.g., (3,3,3) 
                  num_filters=None,
                  activation=activation_default_or_None,
                  init=init_default_or_glorot_uniform,
                  pad=pad_default_or_False,
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=bias_default_or_True,
                  init_bias=init_bias_default_or_0,
                  name=''):
    if len(filter_shape) != 3: 
         raise ValueError('Convolution3D: filter_shape must be a 3D tuple, e.g. (3,3,3)')
    return Convolution(filter_shape, num_filters, activation, init, pad, strides, sharing, bias, init_bias, name=name)

# Deconvolution -- create a deconvolution layer with optional non-linearity
def Deconvolution(filter_shape,        # e.g. (3,3)
                num_filters,
                num_input_filters,
                activation=activation_default_or_None,
                init=init_default_or_glorot_uniform,
                pad=pad_default_or_False,
                strides=1,
                sharing=True,     # (must be True currently)
                lower_pad=(0,),
                upper_pad=(0,),
                bias=bias_default_or_True,
                init_bias=init_bias_default_or_0,
                reduction_rank=1, # (must be 1 currently)
                max_temp_mem_size_in_samples=0, 
                name=''):
    activation = _resolve_activation(activation)
    pad  = pad  if _is_given(pad ) else _get_current_default_options().pad
    bias = bias if _is_given(bias) else _get_current_default_options().bias
    # TODO: there must be a Python trick to do this as a function call on locals or so
    if reduction_rank != 1:
        NotImplementedError("Convolution: reduction_rank other than 1 currently not supported")
    if not sharing:
        NotImplementedError("Convolution: sharing option currently must be True")
    output_channels_shape = _as_tuple(num_filters)
    input_channels_shape = _as_tuple(num_input_filters)
    kernel_shape = output_channels_shape + filter_shape
    param_shape = input_channels_shape + kernel_shape

    filter_rank = len(filter_shape)
    init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
    W = Parameter(param_shape, init=init_kernel, name='W')
    b = Parameter(output_channels_shape + (1,) * len(filter_shape), init=init_bias, name='b') if bias else None

    # expression
    x = Placeholder(name='deconvolution_arg')
    apply_x = convolution (W, x,
                           strides=_as_tuple(strides),
                           sharing=_as_tuple(sharing),
                           auto_padding=_as_tuple(pad),
                           lower_pad=lower_pad,
                           upper_pad=upper_pad,
                           transpose=True,
                           max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
    if bias:
        apply_x = apply_x + b
    apply_x = apply_x >> activation
    return Block(apply_x, 'Deconvolution', name, Record(W=W, b=b), make_block=True)

# Create a Pooling layer with one of following types:
#
#   MaxPooling and GlobalMaxPooling
#   AveragePooling and GlobalAveragePooling
#
# Setting the filter_shape to None, mean global pooling.
from cntk.cntk_py import PoolingType_Max, PoolingType_Average, NDShape
def Pooling(op,      # PoolingType_Max or _Average
            filter_shape,  # e.g. (3,3)
            strides=1,
            pad=False, 
            name=''):
    if op == PoolingType_Average:
        op_name = 'AveragePooling'
        if filter_shape == NDShape.unknown.dimensions(): 
            op_name = 'GlobalAveragePooling' 
    elif op == PoolingType_Max:
        op_name = 'MaxPooling'
        if filter_shape == NDShape.unknown.dimensions(): 
            op_name = 'GlobalMaxPooling'
    else:
        raise ValueError('Pooling: op must be PoolingType_Max or PoolingType_average')

    x = Placeholder(name='pooling_arg')
    apply_x = pooling (x, op, filter_shape, strides=_as_tuple(strides), auto_padding=_as_tuple(pad))
    return Block(apply_x, op_name, name, make_block=True)

# MaxPooling
def MaxPooling(filter_shape,  # e.g. (3,3)
               strides=1,
               pad=False, 
               name=''):
    return Pooling(PoolingType_Max, filter_shape, strides=strides, pad=pad, name=name)

# AveragePooling
def AveragePooling(filter_shape,  # e.g. (3,3)
                   strides=1,
                   pad=False, 
                   name=''):
    return Pooling(PoolingType_Average, filter_shape, strides=strides, pad=pad, name=name)

# GlobalMaxPooling
def GlobalMaxPooling(name=''):
    return Pooling(PoolingType_Max, NDShape.unknown.dimensions(), pad=False, name=name)

# GlobalAveragePooling
def GlobalAveragePooling(name=''):
    return Pooling(PoolingType_Average, NDShape.unknown.dimensions(), pad=False, name=name)

# Create a max unpooling layer
def MaxUnpooling(filter_shape,  # e.g. (3,3)
                 strides=1,
                 pad=False,
                 lower_pad=0,
                 upper_pad=0, 
                 name=''):
    x = Placeholder(name='unpool_arg')
    y = Placeholder(name='pool_arg')
    apply_x = unpooling (x, y, PoolingType_Max, filter_shape, strides=_as_tuple(strides), auto_padding=_as_tuple(pad),
                         lower_pad=_as_tuple(lower_pad), upper_pad=_as_tuple(upper_pad))
    return Block(apply_x, 'MaxUnpooling', name, make_block=True)

# Recurrence() -- run a block recurrently over a time sequence
def Recurrence(over, go_backwards=False, initial_state=initial_state_default_or_None, name=''):
    # helper to compute previous value
    # can take a single Variable/Function or a tuple
    initial_state = initial_state if _is_given(initial_state) else _get_current_default_options().initial_state
    # if initial state is given and a numeric constant, then turn it into a Constant() object
    if np.isscalar(initial_state):
        initial_state = Constant(initial_state, shape=(1)) # TODO: This should be automatically done inside the API.
    def previous_hook(state):
        if isinstance (state, tuple):  # if multiple then apply to each element
            return tuple([previous_hook(s) for s in state])
        # not a tuple: must be a 'scalar', i.e. a single element
        return past_value  (state, initial_state) if not go_backwards else \
               future_value(state, initial_state)
    x = Placeholder(name='recurrence_arg')
    state_forward = over.create_placeholder() # create a placeholder or a tuple of placeholders
    prev_state = previous_hook(state_forward)  # delay (h, c)
    f_x_h_c = over(x, prev_state) # apply the recurrent over
    # this returns a Function (x, (h_prev, c_prev)) -> (h, c)
    h_c = f_x_h_c.outputs
    if type(state_forward) is tuple and len(state_forward) > 1: 
      replacements = { value_forward: value for (value_forward, value) in zip(list(_as_tuple(state_forward)), h_c) }
    else:
      replacements = {(state_forward,)[0] : h_c[0] }
    f_x_h_c.replace_placeholders(replacements)  # resolves state_forward := h_c
    h = f_x_h_c.outputs[0]  # 'h' is a Variable (the output of a Function that computed it)
    if _trace_layers:
        _log_node(h)
        _log_node(combine([h.owner]))
    apply_x = combine([h])     # the Function that yielded 'h', so we get to know its inputs
    # apply_x is a Function x -> h
    return Block(apply_x, 'Recurrence', name, Record(over=over))

# Delay -- delay input
# TODO: This does not really have bound parameters. Should it still be a layer?
def Delay(T=1, initial_state=None, name=''):
    UntestedBranchError("Delay")

    # expression
    x = Placeholder(name='delay_arg')
    if T > 0:
        apply_x = past_value  (x, time_step=T, initial_state=initial_state)
    elif T < 0:
        apply_x = future_value(x, time_step=-T, initial_state=initial_state)
    else:
        apply_x = x
    return Block(apply_x, 'Delay', name, make_block=True)

# Dropout -- create a drop-out layer
def Dropout(prob,name=''):
    # expression
    x = Placeholder(name='dropout_arg')
    apply_x = dropout(x, dropout_rate=prob)
    return Block(apply_x, 'Dropout', name, make_block=True)

# Activation -- create an activation layer 
def Activation(activation=activation_default_or_None, name=''): 
    # expression 
    activation = _resolve_activation(activation)
    x = Placeholder(name='activation_arg') 
    apply_x = activation(x) 
    return Block(apply_x, 'Activation', name, make_block=True) 

# BatchNormalization -- create a batch-normalization layer
# TODO: spatial_rank is broken. We should specify the #slowest-changing axes. E.g. 1 would work for images and vectors. Requires C+ change.
def BatchNormalization(map_rank=None,  # if given then normalize only over this many dimensions. E.g. 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=5000, blend_time_constant=0,
                       epsilon=0.00001, use_cntk_engine=False, 
                       name=''):
    # TODO: make map_rank a default option, once per-layer type defaults are implemented

    # parameters bound to this Function
    norm_shape  = _INFERRED
    if map_rank is not None and map_rank != 1:
        UntestedBranchError("BatchNormalization map_rank can only be 1 or None for now")
    scale        = Parameter(norm_shape, init=init_scale)
    bias         = Parameter(norm_shape, init=0)
    run_mean     = Constant(0, shape=norm_shape)  # note: these are not really constants; they are updated differently
    run_variance = Constant(0, shape=norm_shape)
    run_count    = Constant(0, shape=(1,))

    # expression
    x = Placeholder(name='batch_normalization_arg')
    apply_x = batch_normalization(x, scale, bias, run_mean, run_variance, running_count=run_count, spatial=(map_rank == 1),
                                  normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, 
                                  epsilon=epsilon, use_cudnn_engine=not use_cntk_engine)
    return Block(apply_x, 'BatchNormalization', name, Record(scale=scale, bias=bias, mean=run_mean, variance=run_variance), make_block=True)

# LayerNormalization -- create a layer-normalization layer
def LayerNormalization(initial_scale=1, initial_bias=0, name=''):
    UntestedBranchError("LayerNormalization")

    # parameters bound to this Function
    scale = Parameter((1), init=initial_scale)  # TODO: offer Softplus version for protection, as for Stabilizer
    bias  = Parameter((1), init=initial_bias)

    # expression
    x = Placeholder(name='layer_normalization_arg')
    mean = reduce_mean (x) # normalize w.r.t. actual sample statistics
    x0 = x - mean;
    std = sqrt (reduce_mean (x0 * x0))
    #x_hat = element_divide (x0, std)
    x_hat = x0 / std
    apply_x = x_hat * scale + bias    # denormalize with learned parameters
    return Block(apply_x, 'LayerNormalization', name, Record(scale=scale, bias=bias), make_block=True)
