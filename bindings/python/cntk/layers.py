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
from .ops.functions import Function
from .ops.variables import Variable
from .ops import parameter, input_variable, placeholder_variable, combine
from .ops import times, convolution, pooling, batch_normalization, dropout, splice, sequence
from .utils import Record, _as_tuple
from .blocks import _initializer_for, _INFERRED # init helpers

# import the other pieces of the Layers lib so that users can just use import layers to get the entire Layers lib
from .blocks import *
from .models import *


def Dense(shape, activation=default_override_or(identity), init=default_override_or(glorot_uniform()),
          input_rank=None, map_rank=None,
          bias=default_override_or(True), init_bias=default_override_or(0)):
    '''
    Create a fully-connected linear projection layer with optional non-linear activation.
    Note: shape may describe a tensor as well.
    input_rank given: number of inferred axes to add to W (map_rank must not be given)
    map_rank   given: expand W to leave exactly mapRank axes (input_rank must not be given)
    none       given: expand W to all (same as map_rank=0)
    '''

    activation = get_default_override(Dense, activation=activation)
    init       = get_default_override(Dense, init=init)
    bias       = get_default_override(Dense, bias=bias)
    init_bias  = get_default_override(Dense, init_bias=init_bias)

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
    @Function
    def dense(x):
        r = times(x, W, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)
        if b:
            r = r + b
        if activation is not None:
            r = r >> activation#activation(r)
        return r
    # BUGBUG: the 'out = combine(out, name=f_name)' in Function() messes up the parameter order. Need to fix that first.
    #dense = dense(Placeholder(name='x')) # same as Function() without the combine()

    return Block(dense, 'Dense', Record(W=W, b=b))

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
    @Function
    def embed(x):
        return times(x,E)
    #x = Placeholder(name='embedding_arg')
    #apply_x = times(x, E)
    return Block(embed, 'Embedding', Record(E=E))

# helper to expand a sequence into a window, splicing them along the given axis (which must already exist)
def _window(x, axis, begin, end, initial_state=None):
    shifted = [
        past_value(x, initial_state=initial_state, time_step=-t) if t < 0 else
        x                                                        if t == 0 else
        future_value(x, initial_state=initial_state, time_step=t)
        for t in range(begin, end)
    ]
    return splice(*shifted, axis=axis)

# Convolution -- create a convolution layer with optional non-linearity
#             ( (sample shape) +  (output shape) +  (reduction shape) + (shifting shape)  )
#    in     : ( (sample shape) +                 +  (reduction shape) + (shifting shape)  )
#    kernel : (                +  (output shape) +  (reduction shape) + (rec field shape) )
#    out    : ( (sample shape) +  (output shape) +                    + (shifting shape)  )
# TODO: Can we specify atrous (dilated) convolution? How?
# TODO: sharing = false?
# TODO: conflict of parameter order: filter_shape or num_filters first?
#  - filter_shape first is logical for non-NN applications such as straight image filtering
#  - num_filters first is what Keras does
def Convolution(rf_shape,         # e.g. (3,3)
                num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                sequential=False, # time convolution if True (rf_shape[0] is dynamic axis)
                activation=default_override_or(identity),
                init=default_override_or(glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                sharing=True,     # (must be True currently)
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                transpose=False,  # (must be False currently)
                max_temp_mem_size_in_samples=0):

    activation = get_default_override(Convolution, activation=activation)
    init       = get_default_override(Convolution, init=init)
    pad        = get_default_override(Convolution, pad=pad)
    bias       = get_default_override(Convolution, bias=bias)
    init_bias  = get_default_override(Convolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    rf_shape    = _as_tuple(rf_shape)
    num_filters = _as_tuple(num_filters or ())
    # expand options that can be specified as a single value
    def _pad_to_shape(param):
        param = _as_tuple(param)
        while len(param) < len(rf_shape):
            param = (param[0],) + param
        return param
    strides     = _pad_to_shape(strides)
    sharing     = _pad_to_shape(sharing)
    pad         = _pad_to_shape(pad)

    if reduction_rank > 1:
        raise NotImplementedError("Convolution: reduction_rank other than 0 or 1 currently not supported")
    if transpose:
        raise NotImplementedError("Convolution: transpose option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we fake those dimensions on this level.
    fake_output_depth = num_filters == ()
    fake_input_depth  = reduction_rank == 0
    # 1D convolution is not supported by cudnn, so we also add a fake dimension.
    fake_1D = len(rf_shape) < 2
    #if fake_output_depth or fake_input_depth or fake_1D:
    #    UntestedBranchError("Convolution with depth faking")

    actual_output_channels_shape = num_filters                if not fake_output_depth else (1,)
    actual_reduction_shape       = _INFERRED * reduction_rank if not fake_input_depth  else _INFERRED  # BUGBUG: (1,) crashes
    actual_rf_shape              = (1,) * fake_1D + rf_shape

    # add the dimension to the options as well
    num_faked_axes = fake_input_depth + fake_1D
    strides = (1,)     * num_faked_axes + strides
    sharing = (True,)  * num_faked_axes + sharing
    pad     = (False,) * num_faked_axes + pad

    kernel_shape = actual_reduction_shape + actual_rf_shape # kernel := filter plus reductionDims

    rf_rank = len(rf_shape)

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    from cntk import cntk_py
    if isinstance(init, np.ndarray):
        if reduction_rank != 0:
            raise ValueError("a constant initializer can currently only used without reduction dimension")
        # BUGBUG: ^^ no need. Instead, take whatever reduction dimension is given here as that of the input.
        nominal_W_shape = num_filters + rf_shape
        if init.shape != nominal_W_shape:
            raise ValueError("a constant initializer was passed that is of wrong shape")
        init_kernel = init.reshape(actual_output_channels_shape + kernel_shape) # make it fit
    else:
        init_kernel = _initializer_for(init, Record(filter_rank=rf_rank, output_rank=-len(actual_output_channels_shape)))
        # BUGBUG: It is very confusing that output_rank is negative, esp. since that means count from the start. Solution: add a flag?

    # parameters bound to this Function
    W = Parameter(actual_output_channels_shape + kernel_shape,                init=init_kernel, name='W')                   # (K, C, H, W) aka [ W x H x C x K ]
    b = Parameter(actual_output_channels_shape + (1,) * len(actual_rf_shape), init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]

    # expression
    @Function
    def convolve(x):
        # insert additional axes for various purposes
        sequential_rank = 1 if sequential else 0
        num_inserted_axes = sequential_rank + num_faked_axes
        if num_inserted_axes != 0:
            from .ops import reshape
            x = reshape(x, (1,) * num_inserted_axes, begin_axis=-rf_rank + sequential_rank, end_axis=-rf_rank + sequential_rank) # e.g. (2000, 480, 640) -> (2000, 1, 480, 640)
        # sequential convolution is implemented through explicit stacking for now, since the C++ cannot handle it
        # TODO: if reduction_rank==0 and sequential, we don't need the fake reduction axis, just use the sequential axis instead
        if sequential:
            lpad = (rf_shape[0]-1) // 2  # even frames: take from right; odd frames: symmetric
            x = _window(x, axis=-rf_rank, begin=-lpad, end=-lpad+rf_shape[0], initial_state=None)
        # actual convolution
        # TODO: update the parameter order of convolution() to match the optional ones as in here? (options order matches Keras)
        r = convolution (W, x,
                         strides=strides, sharing=sharing, auto_padding=pad,
                         # TODO: can we rename auto_padding to pad?
                         transpose=transpose,
                         max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
        # if sequential and not padding, then strip the extraneous boundary values
        if sequential and not pad[-rf_rank]:
            from .ops.sequence import slice
            r = slice(r, begin_index=lpad, end_index=-(rf_shape[0]-1-lpad))
        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        # BUGBUG: We still have those axes in the kernel. That can only be solved inside the C++ implementation.
        num_axes_to_remove = sequential + fake_1D + fake_output_depth
        if num_axes_to_remove > 0:
            from .ops import reshape
            r = reshape(r, (), begin_axis=-rf_rank-num_axes_to_remove, end_axis=-rf_rank) # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
        if bias:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r

    return Block(convolve, 'Convolution', Record(W=W, b=b))

# Create a Pooling layer with one of following types:
#
#   MaxPooling and GlobalMaxPooling
#   AveragePooling and GlobalAveragePooling
#
# Setting the filter_shape to None, mean global pooling.
# TODO: add sequential mode like Convolution()
from cntk.cntk_py import PoolingType_Max, PoolingType_Average, NDShape
def _Pooling(op,       # PoolingType_Max or _Average
             rf_shape, # e.g. (3,3)
             sequential=False, # pooling in time if True (rf_shape[0] is dynamic axis)
            strides=1,
            pad=False):

    if sequential:
        raise NotImplementedError("Pooling: sequential option not implemented yet")

    @Function
    def pool(x):
        return pooling (x, op, rf_shape, strides=_as_tuple(strides), auto_padding=_as_tuple(pad))

    if op == PoolingType_Average:
        op_name = 'AveragePooling'
    elif op == PoolingType_Max:
        op_name = 'MaxPooling'
    else:
        raise ValueError('Pooling: op must be PoolingType_Max or PoolingType_average')
    return Block(pool, op_name)

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
# TODO: Can bidirectionality be an option of this? bidirectional=True? What was the reason it cannot?
def Recurrence(over, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False):

    go_backwards  = get_default_override(Recurrence, go_backwards=go_backwards)
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    # TODO: accept 'over' to be a Python function, including CNTK primitives like max().
    #       I.e. run it through Function(); or do those use var-length inputs?
    #       Test by using a recurrence over plus(), but cannot really test because of missing inference.
    # TEST THIS
    import types
    if isinstance(over, types.FunctionType):
        over = Function(over)

    # function that this layer represents
    @Function
    def recurrence(x):

        # TODO: move this entire placeholder business to Function.__call__
        # compute the delayed state variable(s)
        _, *prev_state_args = over.placeholders

        if len(over.outputs) != len(prev_state_args):
            raise TypeError('Recurrence: number of state variables inconsistent between create_placeholder() and recurrent block')
        #out_vars_fwd = [Placeholder(state_var.shape) for state_var in prev_state_args] # create list of placeholders for the state variables
        out_vars_fwd = [Placeholder() for state_var in prev_state_args] # create list of placeholders for the state variables
        # BUGBUG: ^^ This currently leads to "ValueError: Cannot create an NDArrayView using a view shape that has unknown dimensions for any of its axes!"
        # NOPE: works now.

        #print('out_vars_fwd', [p.uid for p in out_vars_fwd])

        # previous function; that is, past or future_value with initial_state baked in
        # BUGBUG: If initial_state itself depends on a Placeholder at this point (e.g. seq2seq), previous_hook will be a binary function...
        # All state variables get delayed with the same function.
        # TODO: Can we use Delay() here?
        def previous_hook(state):
            return past_value  (state, initial_state) if not go_backwards else \
                   future_value(state, initial_state)
        #prev_out_vars = [previous_hook(s) for s in out_vars_fwd]  # delay (state vars)
        prev_out_vars = [Delay(T = -1 if go_backwards else +1, initial_state=initial_state)(s) for s in out_vars_fwd]  # delay (state vars)
        # TODO: allow a learnable initial_state to be passed to each state
        #       Actually, if initial_state is a CNTK Function rather than an initializer, then require to pass it multiple times; otherwise broadcast to all

        #print('prev_out_vars', [p.uid for p in prev_out_vars])

        # apply the recurrent block ('over')
        out = over(x, *prev_out_vars)  # this returns a Function (x, previous outputs...) -> (state vars...)
        #print('over.placeholders', [p.uid for p in over.placeholders])

        # connect the recurrent dependency
        replacements = { var_fwd: var for (var_fwd, var) in zip(out_vars_fwd, list(out.outputs)) }
        #print('out.outputs', [p.uid for p in list(out.outputs)])
        out.replace_placeholders(replacements)  # resolves out_vars_fwd := state_vars

        if not return_full_state:
            out = combine([out.outputs[0]])  # BUGBUG: Without combine(), it fails with "RuntimeError: Runtime exception". TODO: fix this inside Function(lambda)?

        return out

    return Block(recurrence, 'Recurrence', Record(over=over))

# Fold -- final state of a Recurrence()
def Fold(over, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False):

    go_backwards  = get_default_override(Fold, go_backwards=go_backwards)
    initial_state = get_default_override(Fold, initial_state=initial_state)

    # get the scan function
    recurrence = Recurrence(over, go_backwards=go_backwards, initial_state=initial_state, return_full_state=return_full_state)

    # now take the last or first
    select = sequence.first if go_backwards else sequence.last
    fold = recurrence >> tuple(select for output in recurrence.outputs)

    return Block(fold, 'Fold', Record(over=over))

# Delay -- delay input
# TODO: This does not really have bound parameters. Should it still be a layer?
def Delay(T=1, initial_state=default_override_or(0)):
    initial_state = get_default_override(Delay, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    # expression
    @Function
    def delay(x):
        if T > 0:
            return past_value  (x, time_step=T, initial_state=initial_state)
        elif T < 0:
            return future_value(x, time_step=-T, initial_state=initial_state)
        else:
            return x
    return Block(delay, 'Delay')

# Dropout -- create a drop-out layer
def Dropout(prob):
    @Function
    def dropout(x):
        from cntk.ops import dropout # avoid scope mixup
        return dropout(x, dropout_rate=prob)
    return Block(dropout, 'Dropout')

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
    @Function
    def batch_normalize(x):
        #x = Placeholder(name='batch_normalization_arg')
        return batch_normalization(x, scale, bias, run_mean, run_variance, run_count, map_rank == 1, normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, epsilon=epsilon,
                                  use_cudnn_engine=not use_cntk_engine)
    return Block(batch_normalize, 'BatchNormalization', Record(scale=scale, bias=bias, mean=run_mean, variance=run_variance))

# LayerNormalization -- create a layer-normalization layer
# TODO: add an epsilon [CR comment by Nikos]
def LayerNormalization(initial_scale=1, initial_bias=0):
    UntestedBranchError("LayerNormalization")

    # parameters bound to this Function
    scale = Parameter((1), init=initial_scale)  # TODO: offer Softplus version for protection, as for Stabilizer
    bias  = Parameter((1), init=initial_bias)

    # expression
    @Function
    def layer_normalize(x):
        mean = reduce_mean (x) # normalize w.r.t. actual sample statistics
        x0 = x - mean;
        std = sqrt (reduce_mean (x0 * x0))
        #x_hat = element_divide (x0, std)
        x_hat = x0 / std
        return x_hat * scale + bias    # denormalize with learned parameters

    return Block(layer_normalize, 'LayerNormalization', Record(scale=scale, bias=bias))
