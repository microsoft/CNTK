# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# layers -- blocks in the network that are used layer-like, i.e. layered on top of each other
#           e.g. a fully connected layer with non-linearity

# TODO: clean up the dependencies
import numpy as np
from cntk.ops import parameter, input_variable, placeholder_variable, combine
from cntk.ops import times, convolution, pooling, batch_normalization, dropout
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, _as_tuple
from cntk.blocks import *  # TODO: reduce to what we actually use
from cntk.blocks import _trace_layers  # (debugging)

from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# this is what we initialize weight matrices from by default
from cntk.blocks import _initializer_for, _INFERRED

# Dense -- create a fully-connected linear projection layer with optional non-linear activation
# Note: shape may describe a tensor as well.
# input_rank given: number of inferred axes to add to W (map_rank must not be given)
# map_rank   given: expand W to leave exactly mapRank axes (input_rank must not be given)
# none       given: expand W to all (same as map_rank=0)
def Dense(shape, init=default_override_or(glorot_uniform()), activation=default_override_or(identity),
          input_rank=None, map_rank=None,
          bias=default_override_or(True), init_bias=default_override_or(0)):

    init       = get_default_override(Dense, init=init)
    activation = get_default_override(Dense, activation=activation)
    bias       = get_default_override(Dense, bias=bias)
    init_bias  = get_default_override(Dense, init_bias=init_bias)

    #activation = _resolve_activation(activation)
    #bias       = bias if _is_given(bias) else _get_current_default_options().bias
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
    return Block(apply_x, 'Dense', Record(W=W, b=b))

# Embedding -- create a linear embedding layer
# To create an embedding from a file, use this:
#  Embedding(weights=np.load('PATH'))
def Embedding(shape=None, init=default_override_or(glorot_uniform()), weights=None):

    if not is_default_override(init) and weights is not None:
        raise ValueError('Embedding: init and weights options are mutually exclusive')

    init = get_default_override(Embedding, init=init)

    # parameters bound to this Function:
    # no weights given: learn the embedding
    if weights is None:
        if shape is None:
            raise ValueError('Embedding: output shape must be specified')
        #if init is None:
        #    init = init_default_or_glorot_uniform
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
    return Block(apply_x, 'Embedding', Record(E=E))

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
def Convolution(rf_shape,        # e.g. (3,3)
                num_filters=None,    # e.g. 64 or None (which means 1 channel and don't add a dimension_
                activation=default_override_or(identity),
                init=default_override_or(glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                sharing=True,     # (must be True currently)
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1, # (must be 1 currently)
                transpose=False,  # (must be False currently)
                max_temp_mem_size_in_samples=0):

    activation = get_default_override(Convolution, activation=activation)
    init       = get_default_override(Convolution, init=init)
    pad        = get_default_override(Convolution, pad=pad)
    bias       = get_default_override(Convolution, bias=bias)
    init_bias  = get_default_override(Convolution, init_bias=init_bias)
    #activation = _resolve_activation(activation)
    #pad  = pad  if _is_given(pad ) else _get_current_default_options().pad
    #bias = bias if _is_given(bias) else _get_current_default_options().bias

    if reduction_rank != 1:
        NotImplementedError("Convolution: reduction_rank other than 1 currently not supported")
    if transpose:
        NotImplementedError("Convolution: transpose option currently not supported")
    if not sharing:
        NotImplementedError("Convolution: sharing option currently must be True")
    output_channels_shape = _as_tuple(num_filters)
    output_rank = len(output_channels_shape)
    filter_rank = len(rf_shape)
    kernel_shape = _INFERRED * reduction_rank + rf_shape # kernel := filter plus reductionDims

    # parameters bound to this Function
    #init_kernel = glorot_uniform(filter_rank=-filter_rank, output_rank=1)
    init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
    # BUGBUG: It is very confusing that output_rank is negative, esp. since that means count from the start. Solution: add a flag
    W = Parameter(output_channels_shape + kernel_shape,         init=init_kernel, name='W')                   # (K, C, H, W) aka [ W x H x C x K ]
    b = Parameter(output_channels_shape + (1,) * len(rf_shape), init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]

    # expression
    x = Placeholder(name='convolution_arg')
    # TODO: update the parameter order of convolution() to match the optional ones as in here? (options order matches Keras)
    apply_x = convolution (W, x,
                           strides=_as_tuple(strides),
                           sharing=_as_tuple(sharing),
                           auto_padding=_as_tuple(pad),
                           # TODO: can we rename auto_padding to pad?
                           transpose=transpose,
                           max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
    if bias:
        apply_x = apply_x + b
    apply_x = apply_x >> activation
    return Block(apply_x, 'Convolution', Record(W=W, b=b))

# Create a Pooling layer with one of following types:
#
#   MaxPooling and GlobalMaxPooling
#   AveragePooling and GlobalAveragePooling
#
# Setting the filter_shape to None, mean global pooling.
from cntk.cntk_py import PoolingType_Max, PoolingType_Average, NDShape
def _Pooling(op,      # PoolingType_Max or _Average
            rf_shape,  # e.g. (3,3)
            strides=1,
            pad=False):

    x = Placeholder(name='pooling_arg')
    apply_x = pooling (x, op, rf_shape, strides=_as_tuple(strides), auto_padding=_as_tuple(pad))

    if op == PoolingType_Average:
        op_name = 'AveragePooling'
    elif op == PoolingType_Max:
        op_name = 'MaxPooling'
    else:
        raise ValueError('Pooling: op must be PoolingType_Max or PoolingType_average')
    return Block(apply_x, op_name)

# MaxPooling
def MaxPooling(rf_shape,  # e.g. (3,3)
               strides=1,
               pad=default_override_or(False)):
    pad = get_default_override(MaxPooling, pad=pad)
    return _Pooling(PoolingType_Max, rf_shape, strides=strides, pad=pad)

# AveragePooling
def AveragePooling(rf_shape,  # e.g. (3,3)
                   strides=1,
                   pad=default_override_or(False)):
    pad = get_default_override(AveragePooling, pad=pad)
    return _Pooling(PoolingType_Average, rf_shape, strides=strides, pad=pad)

# GlobalMaxPooling
def GlobalMaxPooling():
    return _Pooling(PoolingType_Max, NDShape.unknown.dimensions(), pad=False)

# GlobalAveragePooling
def GlobalAveragePooling():
    return _Pooling(PoolingType_Average, NDShape.unknown.dimensions(), pad=False)

# helper to get the initial_state or the default
def _get_initial_state_or_default(initial_state):
    # TODO: remove this line
    #initial_state = initial_state if _is_given(initial_state) else _get_current_default_options().initial_state
    # if initial state is given and a numeric constant, then turn it into a Constant() object
    if initial_state is None:
        return Constant(0) # note: don't pass None to past_value, because that would default to float32
    elif np.isscalar(initial_state):
        return Constant(initial_state, shape=(1))
    else:
        return initial_state # already in good shape: return as is

# Recurrence() -- run a block recurrently over a time sequence
def Recurrence(over, go_backwards=False, initial_state=default_override_or(0)):

    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    # function that this layer represents
    def recurrence(x):

        # compute the delayed state variable(s)
        # All state variables get delayed with the same function.
        state_forward = over.create_placeholder() # creates list of placeholders for the state variables
        if not isinstance(state_forward, list):
            state_forward = [state_forward]
        def previous_hook(state):
            return past_value  (state, initial_state) if not go_backwards else \
                   future_value(state, initial_state)
        prev_state = [previous_hook(s) for s in state_forward]  # delay (state vars)

        # apply the recurrent block ('over')
        out_and_state = over(x, *prev_state)  # this returns a Function (x, previous state vars...) -> (out, state vars...)
        out, *state_vars = list(out_and_state.outputs)
        if len(state_vars) != len(prev_state):
            raise TypeError('Recurrence: number of state variables inconsistent between create_placeholder() and recurrent block')

        # connect the recurrent dependency
        replacements = { var_forward: var for (var_forward, var) in zip(state_forward, state_vars) }
        out_and_state.replace_placeholders(replacements)  # resolves state_forward := state_vars

        return combine([out])  # BUGBUG: Without this, it fails with "RuntimeError: Runtime exception"

    return Block(recurrence, 'Recurrence', Record(over=over))

# Delay -- delay input
# TODO: This does not really have bound parameters. Should it still be a layer?
def Delay(T=1, initial_state=default_override_or(0)):
    # TODO: change initial_state to a per-function default
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)
    UntestedBranchError("Delay")

    # expression
    x = Placeholder(name='delay_arg')
    if T > 0:
        apply_x = past_value  (x, time_step=T, initial_state=initial_state)
    elif T < 0:
        apply_x = future_value(x, time_step=T, initial_state=initial_state)
    else:
        apply_x = x
    return Block(apply_x, 'Delay')

# Dropout -- create a drop-out layer
def Dropout(prob):
    # expression
    x = Placeholder(name='dropout_arg')
    apply_x = dropout(x, dropout_rate=prob)
    return Block(apply_x, 'Dropout')

# BatchNormalization -- create a batch-normalization layer
# TODO: spatial_rank is broken. We should specify the #slowest-changing axes. E.g. 1 would work for images and vectors. Requires C+ change.
def BatchNormalization(map_rank=default_override_or(None),  # if given then normalize only over this many dimensions. E.g. 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=default_override_or(5000), blend_time_constant=0,
                       epsilon=default_override_or(0.00001), use_cntk_engine=default_override_or(True)):

    map_rank                    = get_default_override(BatchNormalization, map_rank=map_rank)
    normalization_time_constant = get_default_override(BatchNormalization, normalization_time_constant=normalization_time_constant)
    epsilon                     = get_default_override(BatchNormalization, epsilon=epsilon)
    use_cntk_engine             = get_default_override(BatchNormalization, use_cntk_engine=use_cntk_engine)

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
    apply_x = batch_normalization(x, scale, bias, run_mean, run_variance, run_count, map_rank == 1, normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, epsilon=epsilon,
                                  use_cudnn_engine=not use_cntk_engine)
    return Block(apply_x, 'BatchNormalization', Record(scale=scale, bias=bias, mean=run_mean, variance=run_variance))

# LayerNormalization -- create a layer-normalization layer
# TODO: add an epsilon [CR comment by Nikos]
def LayerNormalization(initial_scale=1, initial_bias=0):
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
    return Block(apply_x, 'LayerNormalization', Record(scale=scale, bias=bias))
