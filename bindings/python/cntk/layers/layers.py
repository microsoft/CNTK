# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# layers -- blocks in the network that are used layer-like, i.e. layered on top of each other
#           e.g. a fully connected layer with non-linearity

from __future__ import division
import numpy as np
from ..ops.functions import Function
from ..ops.variables import Variable
from ..ops import parameter, input_variable, placeholder_variable, combine
from ..ops import times, element_times, convolution, pooling, unpooling, batch_normalization, dropout, splice, reshape, sequence, softmax, tanh, reduce_sum, reduce_mean, sqrt
from ..utils import Record, _as_tuple
from .blocks import *
from .blocks import _initializer_for, _get_initial_state_or_default, _INFERRED # helpers


def Dense(shape, activation=default_override_or(identity), init=default_override_or(glorot_uniform()),
          input_rank=None, map_rank=None,
          bias=default_override_or(True), init_bias=default_override_or(0),
          name=''):
    '''
    Layer factory function to create an instance of a fully-connected linear layer of the form
     `activation(input @ W + b)` with weights `W` and bias `b`, and `activation` and `b` being optional.
    `shape` may describe a tensor as well.

    A ``Dense`` layer instance owns its parameter tensors `W` and `b`, and exposes them as attributes ``.W`` and ``.b``.

    Example:
     >>> f = Dense(5, activation=C.relu)
     >>> x = Input(3)
     >>> h = f(x)
     >>> h.shape
         (5,)
     >>> f.W.shape
         (3, 5)
     >>> f.b.value
         array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)

    Args:
     shape ((`int` or `tuple` of `int`s)): vector or tensor dimension of the output of this layer
     activation (:class:`~cntk.ops.functions.Function`, optional): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, default `glorot_uniform()`): initial value of weights `W`
     input_rank (int, optional): number of inferred axes to add to W (`map_rank` must not be given)
     map_rank (int, optional): expand W to leave exactly `map_rank` axes (`input_rank` must not be given)
     bias (boolean, optional, default `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`): initial value of weights `b`
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function` that accepts one input and applies the operation to it
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

    output_rank = len(output_shape)   # support outputs with tensor layouts

    # If input_rank not given then pass a single _INFERRED; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _INFERRED * (input_rank if input_rank is not None else 1)

    if input_rank is not None:
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
    @BlockFunction('Dense', name)
    def dense(x):
        r = times(x, W, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)
        if b:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return dense


def Embedding(shape=None, init=default_override_or(glorot_uniform()), weights=None, name=''):
    '''
    Layer factory function to create a embedding layer.

    An embedding is conceptually a lookup table. For every input token (e.g. a word or any category label), the corresponding
    entry in in the lookup table is returned.

    In CNTK, discrete items such as words are represented as one-hot vectors.
    The table lookup is realized as a matrix product, with a matrix
    whose rows are the embedding vectors.
    Note that multiplying a matrix from the left with a one-hot vector is the same as copying
    out the row for which the input vector is 1.
    CNTK has special optimizations to make this operation as efficient as an actual table lookup if the input is sparse.

    The lookup table in this layer is learnable,
    unless a user-specified one is supplied through the ``weights`` parameter.
    For example, to use an existing embedding table from a file in numpy format, use this:
     ``Embedding(weights=np.load('PATH.npy'))``

    To initialize a learnable lookup table with a given numpy array that is to be used as
    the initial value, pass that array to the ``init`` parameter (not ``weights``).

    An ``Embedding`` instance owns its weight parameter tensor `E`, and exposes it as an attribute ``.E``.

    Example:
     # learnable embedding
     >>> f = Embedding(5)
     >>> x = Input(3)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> f.E.shape
         (3, 5)

     # user-supplied embedding
     >>> f = Embedding(weights=[[.5, .3, .1, .4, .2], [.7, .6, .3, .2, .9]])
     >>> f.E.value
         array([[ 0.5,  0.3,  0.1,  0.4,  0.2],
                [ 0.7,  0.6,  0.3,  0.2,  0.9]], dtype=float32)
     >>> x = Input(2, is_sparse=True)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> e(C.one_hot([[1], [0], [0], [1]], num_classes=2))
     array([[[ 0.7,  0.6,  0.3,  0.2,  0.9]],
     <BLANKLINE>
            [[ 0.5,  0.3,  0.1,  0.4,  0.2]],
     <BLANKLINE>
            [[ 0.5,  0.3,  0.1,  0.4,  0.2]],
     <BLANKLINE>
            [[ 0.7,  0.6,  0.3,  0.2,  0.9]]], dtype=float32)

    Args:
     shape ((`int` or `tuple` of `int`s)): vector or tensor dimension of the output of this layer
     init (scalar or NumPy array or :mod:`cntk.initializer`, default `glorot_uniform()`): (learnable embedding only) initial value of weights `E`
     weights (NumPy array, mutually exclusive with ``init``): (user-supplied embedding only) the lookup table.
      The matrix rows are the embedding vectors, ``weights[i,:]`` being the embedding that corresponds to input category `i`.
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function` that accepts one input and applies the embedding operation to it
    '''

    if not is_default_override(init) and weights is not None:
        raise ValueError('Embedding: init and weights options are mutually exclusive')

    # parameters bound to this Function:
    # no weights given: learn the embedding
    if weights is None:
        if shape is None:
            raise ValueError('Embedding: output shape must be specified')
        init = get_default_override(Embedding, init=init)
        shape = _as_tuple(shape)
        weight_shape = _INFERRED + shape
        E = Parameter(weight_shape, init=init, name='E')
    # weights given: use them as constant
    else:
        import numpy as np
        weights = np.array(weights)
        weight_shape = np.shape(weights)
        if shape is not None: # user may give shape, then it must match
            raise ValueError('Embedding: output shape must not be specified when weights are given')
        E = Constant(weights, name='E')

    # expression
    @BlockFunction('Embedding', name)
    def embed(x):
        return times(x,E)
    return embed


def _window(x, axis, begin, end, step, stride, initial_state=None):
    '''
    helper to expand a sequence into a window, splicing them along the given axis (which must already exist)
    '''
    shifted = [
        past_value(x, initial_state=initial_state, time_step=-t) if t < 0 else
        x                                                        if t == 0 else
        future_value(x, initial_state=initial_state, time_step=t)
        for t in range(begin, end, step)
    ]
    r = splice(*shifted, axis=axis)
    if stride != 1:
        raise NotImplementedError('windowed convolution with stride not yet implemented')
    return r


# helper to expand options that can be specified as a single value
def _pad_to_shape(rf_shape, param, what):
    param = _as_tuple(param)
    if len(param) == 1: # broadcast
        while len(param) < len(rf_shape):
            param = (param[0],) + param
    if len(param) != len(rf_shape):
        raise ValueError("{} parameter ({}) must be a scalar or have same number of elements as the rf_shape parameter ({})".format(what, param, rf_shape))
    return param

# BUGBUG: Can one pass a numpy array as initial values? TODO: add a test case
# Convolution -- create a convolution layer with optional non-linearity
#             ( (sample shape) +  (output shape) +  (reduction shape) + (shifting shape)  )
#    in     : ( (sample shape) +                 +  (reduction shape) + (shifting shape)  )
#    kernel : (                +  (output shape) +  (reduction shape) + (rec field shape) )
#    out    : ( (sample shape) +  (output shape) +                    + (shifting shape)  )
# TODO: Add atrous (dilated) convolution once available.
# TODO: sharing = false?
# TODO: conflict of parameter order: rf_shape or num_filters first?
#  - rf_shape first is logical for non-NN applications such as straight image filtering
#  - num_filters first is what Keras does
# TODO: stride not supported for sequential
def Convolution(rf_shape,         # shape of receptive field, e.g. (3,3)
                num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                sequential=False, # time convolution if True (rf_shape[0] corresponds to dynamic axis)
                activation=default_override_or(identity),
                init=default_override_or(glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                sharing=True,     # (must be True currently)
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                transpose=False,  # (must be False currently)
                max_temp_mem_size_in_samples=0,
                op_name='Convolution', name=''):
    '''
    Layer factory function to create a convolution layer.

    An ``Convolution`` instance owns its weight parameter tensors `W` and `b`, and exposes them as an attributes ``.W`` and ``.b``1.

    Args:
     rf_shape ((`int` or `tuple` of `int`s)): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, optional): number of filters (output feature-map depth). If this parameter is omitted, there will be one filter, but the output shape will have no depth axis.
     sequential (boolean, default `False`): if `True`, also convolve along the dynamic axis. ``rf_shape[0]`` corresponds to dynamic axis.
     activation (:class:`~cntk.ops.functions.Function`, optional): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, default `glorot_uniform()`): initial value of weights `W`
     pad (`bool` or `tuple` of `bool`s, default `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and values outside the valid region will be considered zero.
      Use a `tuple` to spiecfy a per-axis value.
     strides (`int` or `tuple` of `int`s, default `): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to spiecfy a per-axis value.
     bias (boolean, optional, default `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`): initial value of weights `b`
     reduction_rank (`int`, default 1): set to 0 if input has no depth dimension, e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     name (str, optional): the name of the Function instance in the network

    Example:
     # 2D convolution of 5x4 receptive field with output feature-map depth 128:
     >>> f = Convolution((5,4), 128, activation=C.relu)
     >>> x = Input((3,480,640))  # 3-channel color image
     >>> h = f(x)
     >>> h.shape
         (128, 476, 637)
     >>> f.W.shape
         (128, 3, 5, 4)

     # 4D convolution along dynamic axis over a sequence of 2D color images
     >>> from cntk.layers.typing import Sequence, Tensor
     >>> f = Convolution((2,5,4), 128, sequential=True, activation=C.relu) # over 2 consecutive frames
     >>> x = Input(**Sequence[Tensor[3,480,640]])
     >>> h = f(x)
     >>> h.shape
         (128, 476, 637)
     >>> f.W.shape
         (128, 3, 2, 5, 4)

     # 2D convolution over a one-channel black-and-white image, padding, and stride 2 along width dimension
     >>> f = Convolution((3,3), 128, reduction_rank=0, pad=True, strides=(1,2), activation=C.relu)
     >>> x = Input((480,640))
     >>> h = f(x)
     >>> h.shape
         (128, 480, 319)
     >>> f.W.shape
         (128, 1, 3, 3)

    Returns:
        :class:`~cntk.ops.functions.Function` that accepts one input and applies the convolution operation to it
    '''

    activation = get_default_override(Convolution, activation=activation)
    init       = get_default_override(Convolution, init=init)
    pad        = get_default_override(Convolution, pad=pad)
    bias       = get_default_override(Convolution, bias=bias)
    init_bias  = get_default_override(Convolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    rf_shape    = _as_tuple(rf_shape)
    num_filters = _as_tuple(num_filters or ())
    rf_rank = len(rf_shape)
    strides     = _pad_to_shape(rf_shape, strides, 'strides')
    sharing     = _pad_to_shape(rf_shape, sharing, 'sharing')
    pad         = _pad_to_shape(rf_shape, pad, 'pad')

    if reduction_rank > 1:
        raise NotImplementedError("Convolution: reduction_rank other than 0 or 1 currently not supported")
    if transpose:
        raise NotImplementedError("Convolution: transpose option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we emulate those dimensions on this level. TODO: Once this is suppored by the C++ code, remove the emulation here.
    emulating_output_depth = num_filters == ()
    emulating_input_depth  = reduction_rank == 0
    # 1D convolution is not supported by cudnn, so we also add a fake dimension.
    emulating_1D = len(rf_shape) < 2

    actual_output_channels_shape = num_filters                if not emulating_output_depth else (1,)
    actual_reduction_shape       = _INFERRED * reduction_rank if not emulating_input_depth  else _INFERRED  # BUGBUG: (1,) crashes
    actual_rf_shape              = (1,) * emulating_1D + rf_shape

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth + emulating_1D
    strides = (1,)     * num_emulated_axes + strides
    sharing = (True,)  * num_emulated_axes + sharing
    pad     = (False,) * num_emulated_axes + pad

    kernel_shape = actual_reduction_shape + actual_rf_shape # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
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
    @BlockFunction('Convolution', name)
    def convolve(x):
        # insert additional axes for various purposes
        rf_rank_without_seq = rf_rank - sequential    # spatial_shape has rf_rank except if sequential: then first axis of rf_rank belongs to sequential dimension, must subtract
        num_inserted_axes = sequential + num_emulated_axes
        if num_inserted_axes != 0:
            # x: (in_depth, spatial_shape)
            x = reshape(x, (1,) * num_inserted_axes, begin_axis=-rf_rank_without_seq, end_axis=-rf_rank_without_seq if rf_rank_without_seq != 0 else None) # e.g. (2000, 480, 640) -> (2000, 1, 480, 640)
            # x: (in_depth or emulated_in_depth, emulated_1D_extra, seq_rf_shape, spatial_shape)
        # sequential convolution is implemented through explicit stacking for now, since the C++ cannot handle it
        # TODO: if reduction_rank==0 and sequential, we don't need the fake reduction axis, just use the sequential axis instead
        if sequential:
            lpad = (rf_shape[-rf_rank]-1) // 2  # even frames: take from right; odd frames: symmetric
            x = _window(x, axis=-rf_rank, begin=-lpad, end=-lpad+rf_shape[-rf_rank], step=1, stride=strides[-rf_rank], initial_state=None)
        # actual convolution
        r = convolution (W, x,
                         strides=strides, sharing=sharing, auto_padding=pad,
                         # TODO: can we rename auto_padding to pad?
                         transpose=transpose,
                         max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
        # if sequential and not padding, then strip the extraneous boundary values
        if sequential and not pad[-rf_rank]:
            r = sequence.slice(r, begin_index=lpad, end_index=-(rf_shape[-rf_rank]-1-lpad))
        if bias:
            r = r + b
        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        # BUGBUG: We still have those axes in the kernel. That can only be solved inside the C++ implementation.
        num_axes_to_remove = sequential + emulating_1D + emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = reshape(r, (), begin_axis=-rf_rank_without_seq - num_axes_to_remove, end_axis=-rf_rank_without_seq if rf_rank_without_seq != 0 else None) # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
            # (out_depth, spatial_shape)
        if activation is not None:
            r = activation(r)
        return r

    return convolve


# TODO: make sure the xD versions have all the needed parameters
def Convolution1D(rf_shape,         # shape of receptive field, e.g. (3)
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Layer factory function to create a 1D convolution layer with optional non-linearity.
    Same as `Convolution()` except that rf_shape is verified to be 1-dimensional.
    See `Convolution()` for description of parameters.
    '''
    activation = get_default_override(Convolution1D, activation=activation)
    init       = get_default_override(Convolution1D, init=init)
    pad        = get_default_override(Convolution1D, pad=pad)
    bias       = get_default_override(Convolution1D, bias=bias)
    init_bias  = get_default_override(Convolution1D, init_bias=init_bias)
    if len(_as_tuple(rf_shape)) != 1: 
         raise ValueError('Convolution1D: rf_shape must be a scalar')
    return Convolution(rf_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=sharing, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution1D', name=name)


def Convolution2D(rf_shape,         # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Layer factory function to create a 2D convolution layer with optional non-linearity.
    Same as `Convolution()` except that rf_shape is verified to be 2-dimensional.
    See `Convolution()` for description of parameters.
    '''
    activation = get_default_override(Convolution2D, activation=activation)
    init       = get_default_override(Convolution2D, init=init)
    pad        = get_default_override(Convolution2D, pad=pad)
    bias       = get_default_override(Convolution2D, bias=bias)
    init_bias  = get_default_override(Convolution2D, init_bias=init_bias)
    if len(rf_shape) != 2: 
         raise ValueError('Convolution2D: rf_shape must be a 2D tuple, e.g. (3,3)')
    return Convolution(rf_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=sharing, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution2D', name=name)


def Convolution3D(rf_shape,         # shape of receptive field, e.g. (3,3,3). Must be a 3-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  sharing=True,     # (must be True currently)
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Layer factory function to create a 3D convolution layer with optional non-linearity.
    Same as `Convolution()` except that rf_shape is verified to be 3-dimensional.
    See `Convolution()` for description of parameters.
    '''
    activation = get_default_override(Convolution3D, activation=activation)
    init       = get_default_override(Convolution3D, init=init)
    pad        = get_default_override(Convolution3D, pad=pad)
    bias       = get_default_override(Convolution3D, bias=bias)
    init_bias  = get_default_override(Convolution3D, init_bias=init_bias)
    if len(rf_shape) != 3: 
         raise ValueError('Convolution3D: rf_shape must be a 3D tuple, e.g. (3,3,3)')
    return Convolution(rf_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=sharing, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution3D', name=name)


# Deconvolution -- create a deconvolution layer with optional non-linearity
# TODO: need to merge with above. Can it simply be transpose=True?
def Deconvolution(rf_shape,        # shape of receptive field, e.g. (3,3)
                  num_filters,
                  num_input_filters,
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  sharing=True,     # (must be True currently)
                  lower_pad=(0,),
                  upper_pad=(0,),
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (must be 1 currently)
                  max_temp_mem_size_in_samples=0, 
                  name=''):

    '''
    Layer factory function to create a deconvolution layer.
    '''
    #UntestedBranchError("Deconvolution not tested after merge to new Layers lib") # it's actually tested by a end-to-end test

    activation = get_default_override(Deconvolution, activation=activation)
    init       = get_default_override(Deconvolution, init=init)
    pad        = get_default_override(Deconvolution, pad=pad)
    bias       = get_default_override(Deconvolution, bias=bias)
    init_bias  = get_default_override(Deconvolution, init_bias=init_bias)

    if reduction_rank != 1:
        NotImplementedError("Deconvolution: reduction_rank other than 1 currently not supported")
    if not sharing:
        NotImplementedError("Deconvolution: sharing option currently must be True")
    output_channels_shape = _as_tuple(num_filters)
    input_channels_shape = _as_tuple(num_input_filters)
    kernel_shape = output_channels_shape + rf_shape
    param_shape = input_channels_shape + kernel_shape

    filter_rank = len(rf_shape)
    init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
    W = Parameter(param_shape, init=init_kernel, name='W')
    b = Parameter(output_channels_shape + (1,) * len(rf_shape), init=init_bias, name='b') if bias else None

    # expression
    @BlockFunction('Deconvolution', name)
    def deconvolve(x):
        r = convolution (W, x,
                         strides=_as_tuple(strides),
                         sharing=_as_tuple(sharing),
                         auto_padding=_as_tuple(pad),
                         lower_pad=lower_pad,
                         upper_pad=upper_pad,
                         transpose=True,
                         max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
        if bias:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return deconvolve

# TODO: add sequential mode like Convolution()
from cntk.cntk_py import PoolingType_Max, PoolingType_Average, NDShape
def _Pooling(op,       # PoolingType_Max or _Average
             rf_shape, # shape of receptive field, e.g. (3,3)
             sequential=False, # pooling in time if True (rf_shape[0] corresponds to dynamic axis)
             strides=1,
             pad=False,
             op_name=None,
             name=''):
    '''
    Shared part of the various pooling layers.
    Set the rf_shape to None to denote global pooling.
    '''

    if sequential:
        raise NotImplementedError("Pooling: sequential option not implemented yet")

    strides     = _pad_to_shape(rf_shape, strides, 'strides')
    pad         = _pad_to_shape(rf_shape, pad, 'pad')

    @BlockFunction(op_name, name)
    def pool(x):
        return pooling (x, op, rf_shape, strides=strides, auto_padding=pad)
    return pool


def MaxPooling(rf_shape,  # shape of receptive field, e.g. (3,3)
               strides=1,
               pad=default_override_or(False),
               name=''):
    '''
    Layer factory function to create a max-pooling layer.
    '''
    pad = get_default_override(MaxPooling, pad=pad)
    return _Pooling(PoolingType_Max, rf_shape, strides=strides, pad=pad, op_name='MaxPooling', name=name)


def AveragePooling(rf_shape,  # shape of receptive field, e.g. (3,3)
                   strides=1,
                   pad=default_override_or(False),
                   name=''):
    '''
    Layer factory function to create an average-pooling layer.
    '''
    pad = get_default_override(AveragePooling, pad=pad)
    return _Pooling(PoolingType_Average, rf_shape, strides=strides, pad=pad, op_name='AveragePooling', name=name)


# TODO: Is this the same as reduce_max()?
def GlobalMaxPooling(name=''):
    '''
    Layer factory function to create a global max-pooling layer.
    '''
    return _Pooling(PoolingType_Max, NDShape.unknown.dimensions(), pad=False, op_name='GlobalMaxPooling', name=name)


def GlobalAveragePooling(name=''):
    '''
    Layer factory function to create a global average-pooling layer.
    '''
    return _Pooling(PoolingType_Average, NDShape.unknown.dimensions(), pad=False, op_name='GlobalAveragePooling', name=name)


# Create a max unpooling layer
# TODO: merge this. Test: Tests\EndToEndTests\CNTKv2Python\Examples\deconv_MNIST_test.py, Tests\EndToEndTests\Examples\Image\GettingStarted\07_Deconvolution
def MaxUnpooling(rf_shape,  # shape of receptive field, e.g. (3,3)
                 strides=1,
                 pad=False,
                 lower_pad=0,
                 upper_pad=0, 
                 name=''):

    strides     = _pad_to_shape(rf_shape, strides, 'strides')
    pad         = _pad_to_shape(rf_shape, pad, 'pad')

    @BlockFunction('MaxUnpooling', name)
    def maxunpool(x, y):
        return unpooling (x, y, PoolingType_Max, rf_shape, strides=strides, auto_padding=pad,
                         lower_pad=_as_tuple(lower_pad), upper_pad=_as_tuple(upper_pad))
    return maxunpool


# TODO: call out that prob is 1-prob in TF
def Dropout(prob, name=''):
    '''
    Layer factory function to create a drop-out layer.
    '''
    @BlockFunction('Dropout', name)
    def dropout_f(x):
        return dropout(x, dropout_rate=prob)
    return dropout_f


def Activation(activation=default_override_or(identity), name=''): 
    '''
    Layer factory function to create an activation layer.
    Activation functions can be used directly in CNTK, so there is no difference between
    ``y = relu(x)`` and ``y = Activation(relu)(x)``.
    This layer is useful if one wants to configure the activation function
    with ``default_options``, or when its invocation should be named.
    '''
    activation = get_default_override(Activation, activation=activation)
    @BlockFunction('Activation', name)
    def activation_f(x):
        return activation(x) 
    return activation_f


# TODO: map_rank is broken. We should specify the #slowest-changing axes. E.g. 1 would work for images and vectors. Requires C++ change.
def BatchNormalization(map_rank=default_override_or(None),  # if given then normalize only over this many dimensions. E.g. 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=default_override_or(5000), blend_time_constant=0,
                       epsilon=default_override_or(0.00001), use_cntk_engine=default_override_or(False),
                       name=''):
    '''
    Layer factory function to create a batch-normalization layer.
    '''

    map_rank                    = get_default_override(BatchNormalization, map_rank=map_rank)
    normalization_time_constant = get_default_override(BatchNormalization, normalization_time_constant=normalization_time_constant)
    epsilon                     = get_default_override(BatchNormalization, epsilon=epsilon)
    use_cntk_engine             = get_default_override(BatchNormalization, use_cntk_engine=use_cntk_engine)

    # parameters bound to this Function
    norm_shape  = _INFERRED
    if map_rank is not None and map_rank != 1:
        UntestedBranchError("BatchNormalization map_rank can only be 1 or None for now")
    scale        = Parameter(norm_shape, init=init_scale, name='scale')
    bias         = Parameter(norm_shape, init=0,          name='bias')
    run_mean     = Constant(0, shape=norm_shape, name='mean')  # note: these are not really constants; they are updated differently
    run_variance = Constant(0, shape=norm_shape, name='variance')
    run_count    = Constant(0, shape=(),         name='count')

    # expression
    @BlockFunction('BatchNormalization', name)
    def batch_normalize(x):
        return batch_normalization(x, scale, bias, run_mean, run_variance, running_count=run_count,
                                   spatial=map_rank == 1, normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, epsilon=epsilon,
                                   use_cudnn_engine=not use_cntk_engine)

    return batch_normalize

def LayerNormalization(initial_scale=1, initial_bias=0, epsilon=default_override_or(0.00001), name=''):
    '''
    Layer factory function to create a function that implements layer normalization.
    '''
    #UntestedBranchError("LayerNormalization")
    epsilon = get_default_override(LayerNormalization, epsilon=epsilon)

    # parameters bound to this Function
    scale = Parameter((1), init=initial_scale, name='scale')  # TODO: offer Softplus version for protection, as for Stabilizer
    bias  = Parameter((1), init=initial_bias,  name='bias')

    # expression
    @BlockFunction('LayerNormalization', name)
    def layer_normalize(x):
        mean = reduce_mean(x) # normalize w.r.t. actual sample statistics
        x0 = x - mean;
        std = sqrt (reduce_mean (x0 * x0))
        if (epsilon != 0):
            std += epsilon
        x_hat = x0 / std
        return x_hat * scale + bias    # denormalize with learned parameters
    return layer_normalize


def Label(name):
    '''
    Layer factory function to create a dummy layer with a given name.
    This can be used to access an intermediate value flowing through computation. E.g.
      ``model = Dense(...) >> Label('hidden') >> Dense(...)
        intermediate_val = model.hidden``
    '''
    @Function  # cannot be a BlockFunction since that would hide the label
    def label(x):
        return alias(x, name=name)
    return label
