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
from ..variables import Variable, Record, Constant
from ..ops import parameter, input, placeholder, combine
from ..ops import times, element_times, convolution, convolution_transpose, pooling, unpooling, batch_normalization, dropout, splice, reshape, sequence, softmax, tanh, reduce_sum, reduce_mean, sqrt
from cntk.internal import _as_tuple
from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as SentinelValueForAutoSelectRandomSeed
from .blocks import *
from .higher_order_layers import *
from .blocks import _initializer_for, _get_initial_state_or_default, _INFERRED # helpers


def Dense(shape, activation=default_override_or(identity), init=default_override_or(glorot_uniform()),
          input_rank=None, map_rank=None,
          bias=default_override_or(True), init_bias=default_override_or(0),
          name=''):
    '''
    Dense(shape, activation=identity, init=glorot_uniform(), input_rank=None, map_rank=None, bias=True, init_bias=0, name='')

    Layer factory function to create an instance of a fully-connected linear layer of the form
    `activation(input @ W + b)` with weights `W` and bias `b`, and `activation` and `b` being optional.
    `shape` may describe a tensor as well.

    A ``Dense`` layer instance owns its parameter tensors `W` and `b`, and exposes them as attributes ``.W`` and ``.b``.

    Example:
     >>> f = Dense(5, activation=C.relu)
     >>> x = input(3)
     >>> h = f(x)
     >>> h.shape
         (5,)
     >>> f.W.shape
         (3, 5)
     >>> f.b.value
         array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)

     >>> # activation through default options
     >>> with default_options(activation=C.relu):
     ...     f = Dense(500)

    Args:
     shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
     activation (:class:`~cntk.ops.functions.Function`, defaults to identity): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     input_rank (int, defaults to `None`): number of inferred axes to add to W (`map_rank` must not be given)
     map_rank (int, defaults to `None`): expand W to leave exactly `map_rank` axes (`input_rank` must not be given)
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defualts to 0): initial value of weights `b`
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function: 
        A function that accepts one argument and applies the operation to it
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
    Embedding(shape=None, init=glorot_uniform(), weights=None, name='')

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
    For example, to use an existing embedding table from a file in numpy format, use this::

      Embedding(weights=np.load('PATH.npy'))

    To initialize a learnable lookup table with a given numpy array that is to be used as
    the initial value, pass that array to the ``init`` parameter (not ``weights``).

    An ``Embedding`` instance owns its weight parameter tensor `E`, and exposes it as an attribute ``.E``.

    Example:
     >>> # learnable embedding
     >>> f = Embedding(5)
     >>> x = input(3)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> f.E.shape
         (3, 5)

     >>> # user-supplied embedding
     >>> f = Embedding(weights=[[.5, .3, .1, .4, .2], [.7, .6, .3, .2, .9]])
     >>> f.E.value
         array([[ 0.5,  0.3,  0.1,  0.4,  0.2],
                [ 0.7,  0.6,  0.3,  0.2,  0.9]], dtype=float32)
     >>> x = input(2, is_sparse=True)
     >>> e = f(x)
     >>> e.shape
         (5,)
     >>> e(C.Value.one_hot([[1], [0], [0], [1]], num_classes=2))
     array([[ 0.7,  0.6,  0.3,  0.2,  0.9],
            [ 0.5,  0.3,  0.1,  0.4,  0.2],
            [ 0.5,  0.3,  0.1,  0.4,  0.2],
            [ 0.7,  0.6,  0.3,  0.2,  0.9]], dtype=float32)

    Args:
     shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): (learnable embedding only) initial value of weights `E`
     weights (NumPy array, mutually exclusive with ``init``, defuats to `None`): (user-supplied embedding only) the lookup table.
      The matrix rows are the embedding vectors, ``weights[i,:]`` being the embedding that corresponds to input category `i`.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the embedding operation to it
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
def _pad_to_shape(filter_shape, param, what):
    param = _as_tuple(param)
    if len(param) == 1: # broadcast
        while len(param) < len(filter_shape):
            param = (param[0],) + param
    if len(param) != len(filter_shape):
        raise ValueError("{} parameter ({}) must be a scalar or have same number of elements as the filter_shape parameter ({})".format(what, param, filter_shape))
    return param

# BUGBUG: Can one pass a numpy array as initial values? TODO: add a test case
# Convolution -- create a convolution layer with optional non-linearity
#             ( (sample shape) +  (output shape) +  (reduction shape) + (spatial shape)   )
#    in     : ( (sample shape) +                 +  (reduction shape) + (spatial shape)   )
#    kernel : (                +  (output shape) +  (reduction shape) + (rec field shape) )
#    out    : ( (sample shape) +  (output shape) +                    + (spatial shape)   )
# TODO: Add atrous (dilated) convolution once available.
# TODO: sharing = false? I'd need that for speech feature extraction.
# TODO: should we allow to pass fixed weights instead? Like for Embedding? E.g. audio filters
# TODO: this is not a convolution but a correlation, and W's shape has input and output depth reverted.
#       Transposition of the weight matrix would do the right thing for both cases. Should we default to correctness, i.e. transpose_weight?
# TODO: conflict of parameter order: filter_shape or num_filters first?
#  - filter_shape first is logical for non-NN applications such as straight image filtering
#  - num_filters first is what Keras does
# TODO: stride not supported for sequential
def Convolution(filter_shape,     # shape of receptive field, e.g. (3,3)
                num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                sequential=False, # time convolution if True (filter_shape[0] corresponds to dynamic axis)
                activation=default_override_or(identity),
                init=default_override_or(glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                sharing=True,     # (must be True currently)
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)  --TODO: call it item_rank?
                transpose_weight=False,  # (must be False currently)
                max_temp_mem_size_in_samples=0,
                op_name='Convolution', name=''):
    '''
    Convolution(filter_shape, num_filters=None, sequential=False, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, reduction_rank=1, transpose_weight=False, max_temp_mem_size_in_samples=0, op_name='Convolution', name='')

    Layer factory function to create a convolution layer.

    This implements a convolution operation over items arranged on an N-dimensional grid, such as pixels in an image.
    Typically, each item is a vector (e.g. pixel: R,G,B), and the result is, in turn, a vector.
    The item-grid dimensions are referred to as the *spatial* dimensions (e.g. dimensions of an image),
    while the vector dimension of the individual items is often called *feature-map depth*.

    For each item, convolution gathers a window ("receptive field") of items surrounding the item's position on the grid,
    and applies a little fully-connected network to it (the same little network is applied to all item positions).
    The size (spatial extent) of the receptive field is given by ``filter_shape``.
    E.g. to specify a 2D convolution, ``filter_shape`` should be a tuple of two integers, such as `(5,5)`;
    an example for a 3D convolution (e.g. video or an MRI scan) would be ``filter_shape=(3,3,3)``;
    while for a 1D convolution (e.g. audio or text), ``filter_shape`` has one element, such as (3,) or just 3.

    The dimension of the input items (input feature-map depth) is not to be specified. It is known from the input.
    The dimension of the output items (output feature-map depth) generated for each item position is given by ``num_filters``.

    If the input is a sequence, the sequence elements are by default treated independently.
    To convolve along the sequence dimension as well, pass ``sequential=True``.
    This is useful for variable-length inputs, such as video
    or natural-language processing (word n-grams).
    Note, however, that convolution does not support sparse inputs.

    Both input and output items can be scalars intead of vectors. For scalar-valued input items,
    such as pixels on a black-and-white image, or samples of an audio clip, specify ``reduction_rank=0``.
    If the output items are scalar, pass ``num_filters=()`` or ``None``.

    A ``Convolution`` instance owns its weight parameter tensors `W` and `b`, and exposes them as an attributes ``.W`` and ``.b``.
    The weights will have the shape ``(num_filters, input_feature_map_depth, *filter_shape)``

    Example:
     >>> # 2D convolution of 5x4 receptive field with output feature-map depth 128:
     >>> f = Convolution((5,4), 128, activation=C.relu)
     >>> x = input((3,480,640))  # 3-channel color image
     >>> h = f(x)
     >>> h.shape
         (128, 476, 637)
     >>> f.W.shape  # will have the form (num_filters, input_depth, *filter_shape)
         (128, 3, 5, 4)

     >>> # 2D convolution over a one-channel black-and-white image, padding, and stride 2 along width dimension
     >>> f = Convolution((3,3), 128, reduction_rank=0, pad=True, strides=(1,2), activation=C.relu)
     >>> x = input((480,640))
     >>> h = f(x)
     >>> h.shape
         (128, 480, 320)
     >>> f.W.shape
         (128, 1, 3, 3)

     >>> # 3D convolution along dynamic axis over a sequence of 2D color images
     >>> from cntk.layers.typing import Sequence, Tensor
     >>> f = Convolution((2,5,4), 128, sequential=True, activation=C.relu) # over 2 consecutive frames
     >>> x = input(**Sequence[Tensor[3,480,640]])  # a variable-length video of 640x480 RGB images
     >>> h = f(x)
     >>> h.shape   # this is the shape per video frame: 637x476 activation vectors of length 128 each
         (128, 476, 637)
     >>> f.W.shape # (output featuer map depth, input depth, and the three filter dimensions)
         (128, 3, 2, 5, 4)

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     sequential (bool, defaults to `False`): if `True`, also convolve along the dynamic axis. ``filter_shape[0]`` corresponds to dynamic axis.
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     sharing (bool, defaults to `True`): When `True`, every position uses the same Convolution kernel.  When `False`, you can have a different Convolution kernel per position, but `False` is not supported.
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     transpose_weight (bool, defaults to `False`): When this is `True` this is convolution, otherwise this is correlation (which is common for most toolkits)
     max_temp_mem_size_in_samples (int, defaults to 0): Limits the amount of memory for intermiadate convolution results.  A value of 0 means, memory is automatically managed.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it
    '''

    activation = get_default_override(Convolution, activation=activation)
    init       = get_default_override(Convolution, init=init)
    pad        = get_default_override(Convolution, pad=pad)
    bias       = get_default_override(Convolution, bias=bias)
    init_bias  = get_default_override(Convolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    filter_shape = _as_tuple(filter_shape)
    num_filters  = _as_tuple(num_filters or ())
    filter_rank  = len(filter_shape)
    strides      = _pad_to_shape(filter_shape, strides, 'strides')
    sharing      = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad          = _pad_to_shape(filter_shape, pad, 'pad')

    if reduction_rank > 1:
        raise NotImplementedError("Convolution: reduction_rank other than 0 or 1 currently not supported")
    if transpose_weight:
        raise NotImplementedError("Convolution: transpose_weight option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we emulate those dimensions on this level. TODO: Once this is suppored by the C++ code, remove the emulation here.
    emulating_output_depth = num_filters == ()
    emulating_input_depth  = reduction_rank == 0
    # 1D convolution is not supported by cudnn, so we also add a fake dimension.
    emulating_1D = len(filter_shape) < 2

    actual_output_channels_shape = num_filters                if not emulating_output_depth else (1,)
    actual_reduction_shape       = _INFERRED * reduction_rank if not emulating_input_depth  else _INFERRED  # BUGBUG: (1,) crashes
    actual_filter_shape          = (1,) * emulating_1D + filter_shape

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth + emulating_1D
    strides = (1,)     * num_emulated_axes + strides
    sharing = (True,)  * num_emulated_axes + sharing
    pad     = (False,) * num_emulated_axes + pad

    kernel_shape = actual_reduction_shape + actual_filter_shape # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    if isinstance(init, np.ndarray):
        if reduction_rank != 0:
            raise ValueError("a constant initializer can currently only used without reduction dimension")
        # BUGBUG: ^^ no need. Instead, take whatever reduction dimension is given here as that of the input.
        nominal_W_shape = num_filters + filter_shape
        if init.shape != nominal_W_shape:
            raise ValueError("a constant initializer was passed that is of wrong shape")
        init_kernel = init.reshape(actual_output_channels_shape + kernel_shape) # make it fit
    else:
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))
        # BUGBUG: It is very confusing that output_rank is negative, esp. since that means count from the start. Solution: add a flag?

    # parameters bound to this Function
    W = Parameter(actual_output_channels_shape + kernel_shape,                    init=init_kernel, name='W')                   # (K, C, H, W) aka [ W x H x C x K ]
    b = Parameter(actual_output_channels_shape + (1,) * len(actual_filter_shape), init=init_bias,   name='b') if bias else None # (K,    1, 1) aka [ 1 x 1 x     K ]

    # TODO: Should we cater to the special case of 1D convolution for text? I.e. sequential only (filter_shape=()).
    #       In that case, the convolution is the embedding, and we should use a matrix product to support sparse inputs.
    #       Or add sparse support to splice().

    # expression
    @BlockFunction('Convolution', name)
    def convolve(x):
        # insert additional axes for various purposes
        filter_rank_without_seq = filter_rank - sequential    # spatial_shape has filter_rank except if sequential: then first axis of filter_rank belongs to sequential dimension, must subtract
        num_inserted_axes = sequential + num_emulated_axes
        if num_inserted_axes != 0:
            # x: (in_depth, spatial_shape)
            x = reshape(x, (1,) * num_inserted_axes,    # e.g. (2000, 480, 640) -> (2000, 1, 480, 640)
                        begin_axis=-filter_rank_without_seq if filter_rank_without_seq != 0 else Axis.new_leading_axis(),
                        end_axis  =-filter_rank_without_seq if filter_rank_without_seq != 0 else None)
            # x: (in_depth or emulated_in_depth, emulated_1D_extra, seq_filter_shape, spatial_shape)
        # sequential convolution is implemented through explicit stacking for now, since the C++ cannot handle it
        # TODO: if reduction_rank==0 and sequential, we don't need the fake reduction axis, just use the sequential axis instead
        if sequential:
            lpad = (filter_shape[-filter_rank]-1) // 2  # even frames: take from right; odd frames: symmetric
            x = _window(x, axis=-filter_rank, begin=-lpad, end=-lpad+filter_shape[-filter_rank], step=1, stride=strides[-filter_rank], initial_state=None)
        # actual convolution
        sequential_emulated_axis = len(pad) - filter_rank if sequential else None # static-axis convolution must not pad the simulated sequential dimension (it must reduce to 1)
        r = convolution (W, x,
                         strides=strides, sharing=sharing, auto_padding=tuple(p if i != sequential_emulated_axis else False for i, p in enumerate(pad)),
                         # TODO: can we rename auto_padding to pad?
                         max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
        # if sequential and not padding, then strip the extraneous boundary values
        if sequential and not pad[-filter_rank]:
            r = sequence.slice(r, begin_index=lpad, end_index=-(filter_shape[-filter_rank]-1-lpad))
        if bias:
            r = r + b
        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        # BUGBUG: We still have those axes in the kernel. That can only be solved inside the C++ implementation.
        num_axes_to_remove = sequential + emulating_1D + emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = reshape(r, (),    # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
                        begin_axis=-filter_rank_without_seq - num_axes_to_remove,  # no need for Axis.new_leading_axis() since expression < 0 guaranteed
                        end_axis  =-filter_rank_without_seq if filter_rank_without_seq != 0 else None)
            # (out_depth, spatial_shape)
        if activation is not None:
            r = activation(r)
        return r

    return convolve


# TODO: make sure the xD versions have all the needed parameters
def Convolution1D(filter_shape,     # shape of receptive field, e.g. (3)
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Convolution1D(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, reduction_rank=1, name='')

    Layer factory function to create a 1D convolution layer with optional non-linearity.
    Same as `Convolution()` except that filter_shape is verified to be 1-dimensional.
    See `Convolution()` for extensive documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it

    '''
    
    activation = get_default_override(Convolution1D, activation=activation)
    init       = get_default_override(Convolution1D, init=init)
    pad        = get_default_override(Convolution1D, pad=pad)
    bias       = get_default_override(Convolution1D, bias=bias)
    init_bias  = get_default_override(Convolution1D, init_bias=init_bias)
    if len(_as_tuple(filter_shape)) != 1: 
         raise ValueError('Convolution1D: filter_shape must be a scalar')
    return Convolution(filter_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=True, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution1D', name=name)


def Convolution2D(filter_shape,     # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Convolution2D(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, reduction_rank=1, name='')

    Layer factory function to create a 2D convolution layer with optional non-linearity.
    Same as `Convolution()` except that filter_shape is verified to be 2-dimensional.
    See `Convolution()` for extensive documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it

    '''

    activation = get_default_override(Convolution2D, activation=activation)
    init       = get_default_override(Convolution2D, init=init)
    pad        = get_default_override(Convolution2D, pad=pad)
    bias       = get_default_override(Convolution2D, bias=bias)
    init_bias  = get_default_override(Convolution2D, init_bias=init_bias)
    if len(_as_tuple(filter_shape)) > 2: 
         raise ValueError('Convolution2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)')
    filter_shape = _pad_to_shape((0,0), filter_shape, 'filter_shape')
    return Convolution(filter_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=True, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution2D', name=name)


def Convolution3D(filter_shape,     # shape of receptive field, e.g. (3,3,3). Must be a 3-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(glorot_uniform()),
                  pad=default_override_or(False),
                  strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  name=''):
    '''
    Convolution3D(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, reduction_rank=1, name='')

    Layer factory function to create a 3D convolution layer with optional non-linearity.
    Same as `Convolution()` except that filter_shape is verified to be 3-dimensional.
    See `Convolution()` for extensive documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it

    '''

    activation = get_default_override(Convolution3D, activation=activation)
    init       = get_default_override(Convolution3D, init=init)
    pad        = get_default_override(Convolution3D, pad=pad)
    bias       = get_default_override(Convolution3D, bias=bias)
    init_bias  = get_default_override(Convolution3D, init_bias=init_bias)
    if len(_as_tuple(filter_shape)) > 3: 
         raise ValueError('Convolution3D: filter_shape must be a scalar or a 3D tuple, e.g. 3 or (3,3,3)')
    filter_shape = _pad_to_shape((0,0,0), filter_shape, 'filter_shape')
    return Convolution(filter_shape, num_filters=num_filters, activation=activation, init=init, pad=pad, strides=strides, sharing=True, bias=bias, init_bias=init_bias, reduction_rank=reduction_rank, op_name='Convolution3D', name=name)


# ConvolutionTranspose -- create a deconvolution layer with optional non-linearity
# TODO: need to merge with above. Can it simply be transpose=True?
def ConvolutionTranspose(filter_shape,        # shape of receptive field, e.g. (3,3)
                         num_filters,
                         activation=default_override_or(identity),
                         init=default_override_or(glorot_uniform()),
                         pad=default_override_or(False),
                         strides=1,
                         sharing=True,     # (must be True currently)
                         bias=default_override_or(True),
                         init_bias=default_override_or(0),
                         output_shape=None, 
                         reduction_rank=1, # (must be 1 currently)
                         max_temp_mem_size_in_samples=0, 
                         name=''):

    '''
    ConvolutionTranspose(filter_shape, num_filters, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, output_shape=None, reduction_rank=1, max_temp_mem_size_in_samples=0, name='')

    Layer factory function to create a convolution transpose layer.

    This implements a convolution_transpose operation over items arranged on an N-dimensional grid, such as pixels in an image.
    Typically, each item is a vector (e.g. pixel: R,G,B), and the result is, in turn, a vector.
    The item-grid dimensions are referred to as the *spatial* dimensions (e.g. dimensions of an image),
    while the vector dimensions of the individual items are often called *feature-map depth*.

    Convolution transpose is also known as ``fractionally strided convolutional layers``, or, ``deconvolution``. 
    This operation is used in image and language processing applications. It supports arbitrary
    dimensions, strides, and padding. 

    The forward and backward computation of convolution transpose is the inverse of convolution. That is, during forward
    pass the input layer's items are spread into the output same as the backward spread of gradients in convolution. The 
    backward pass, on the other hand, performs a convolution same as the forward pass of convolution. 

    The size (spatial extent) of the receptive field for convolution transpose is given by ``filter_shape``.
    E.g. to specify a 2D convolution transpose, ``filter_shape`` should be a tuple of two integers, such as `(5,5)`;
    an example for a 3D convolution transpose (e.g. video or an MRI scan) would be ``filter_shape=(3,3,3)``;
    while for a 1D convolution transpose (e.g. audio or text), ``filter_shape`` has one element, such as (3,).

    The dimension of the input items (feature-map depth) is not specified, but known from the input.
    The dimension of the output items generated for each item position is given by ``num_filters``.

    A ``ConvolutionTranspose`` instance owns its weight parameter tensors `W` and `b`, and exposes them as an attributes ``.W`` and ``.b``.
    The weights will have the shape ``(input_feature_map_depth, num_filters, *filter_shape)``. 

    Example:
     >>> # 2D convolution transpose of 3x4 receptive field with output feature-map depth 128:
     >>> f = ConvolutionTranspose((3,4), 128, activation=C.relu)
     >>> x = input((3,480,640))  # 3-channel color image
     >>> h = f(x)
     >>> h.shape
         (128, 482, 643)
     >>> f.W.shape  # will have the form (input_depth, num_filters, *filter_shape)
         (3, 128, 3, 4)

    Args:
     filter_shape (`int` or tuple of `int`\ s): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, optional): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, default :func:`~cntk.initializer.glorot_uniform`): initial value of weights `W`
     pad (`bool` or tuple of `bool`\ s, default `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or tuple of `int`\ s, default 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     sharing (`bool`, default `True`): weight sharing, must be True for now. 
     bias (`bool`, optional, default `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`): initial value of weights `b`
     output_shape (`int` or tuple of `int`\ s): output shape. When strides > 2, the output shape is non-deterministic. User can specify the wanted output shape. Note the 
      specified shape must satisify the condition that if a convolution is perform from the output with the same setting, the result must have same shape as the input. 
     reduction_rank (`int`, default 1): must be 1 for now. 
      that is stored with tensor shape (H,W) instead of (1,H,W)
     max_temp_mem_size_in_samples (`int`, default 0): set to a positive number to define the maximum workspace memory for convolution. 
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function` that accepts one argument and applies the convolution operation to it
    '''

    activation = get_default_override(ConvolutionTranspose, activation=activation)
    init       = get_default_override(ConvolutionTranspose, init=init)
    pad        = get_default_override(ConvolutionTranspose, pad=pad)
    bias       = get_default_override(ConvolutionTranspose, bias=bias)
    init_bias  = get_default_override(ConvolutionTranspose, init_bias=init_bias)
    output_shape = get_default_override(ConvolutionTranspose, output_shape=output_shape)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    filter_shape = _as_tuple(filter_shape)
    num_filters  = _as_tuple(num_filters)
    filter_rank  = len(filter_shape)
    strides      = _pad_to_shape(filter_shape, strides, 'strides')
    sharing      = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad          = _pad_to_shape(filter_shape, pad, 'pad')

    if reduction_rank != 1:
        NotImplementedError("ConvolutionTranspose: reduction_rank other than 1 currently not supported")
    if not sharing:
        NotImplementedError("ConvolutionTranspose: sharing option currently must be True")
    output_channels_shape = _as_tuple(num_filters)
    kernel_shape = _INFERRED * reduction_rank + filter_shape # kernel := filter plus reductionDims  
    if output_shape is None:  
        kernel_shape = output_channels_shape + filter_shape 
    param_shape = _INFERRED * reduction_rank + kernel_shape

    output_full_shape = output_shape 
    if output_shape is not None:
        output_full_shape = output_channels_shape + output_shape 

    filter_rank = len(filter_shape)
    init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-1))
    W = Parameter(param_shape, init=init_kernel, name='W')
    b = Parameter(output_channels_shape + (1,) * len(filter_shape), init=init_bias, name='b') if bias else None

    # expression
    @BlockFunction('ConvolutionTranspose', name)
    def convolve_transposed(x):
        r = convolution_transpose(W, x,
                                  strides=_as_tuple(strides),
                                  sharing=_as_tuple(sharing),
                                  auto_padding=_as_tuple(pad),
                                  output_shape=output_full_shape, 
                                  max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)
        if bias:
            r = r + b
        if activation is not None:
            r = activation(r)
        return r
    return convolve_transposed

# ConvolutionTranspose1D -- create a 1D convolution transpose layer with optional non-linearity
def ConvolutionTranspose1D(filter_shape,        # a scalar, e.g., 3 
                           num_filters,
                           activation=default_override_or(identity),
                           init=default_override_or(glorot_uniform()),
                           pad=default_override_or(False),
                           strides=1,
                           bias=default_override_or(True),
                           init_bias=default_override_or(0),
                           output_shape=None, 
                           name=''):
    '''
    ConvolutionTranspose1D(filter_shape, num_filters, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, output_shape=None, name='')

    Layer factory function to create a 1D convolution transpose layer with optional non-linearity.
    Same as `ConvolutionTranspose()` except that filter_shape is verified to be 1-dimensional.
    See `ConvolutionTranspose()` for extensive documentation.
    '''
    activation = get_default_override(ConvolutionTranspose1D, activation=activation)
    init       = get_default_override(ConvolutionTranspose1D, init=init)
    pad        = get_default_override(ConvolutionTranspose1D, pad=pad)
    bias       = get_default_override(ConvolutionTranspose1D, bias=bias)
    init_bias  = get_default_override(ConvolutionTranspose1D, init_bias=init_bias)
    output_shape = get_default_override(ConvolutionTranspose1D, output_shape=output_shape)
    if len(_as_tuple(filter_shape)) != 1: 
         raise ValueError('ConvolutionTranspose1D: filter_shape must be a scalar')
    return ConvolutionTranspose(filter_shape, num_filters, activation, init, pad, strides, True, bias, init_bias, output_shape, name=name)

# ConvolutionTranspose2D -- create a 2D convolution transpose layer with optional non-linearity
def ConvolutionTranspose2D(filter_shape,        # a 2D tuple, e.g., (3,3) 
                           num_filters,
                           activation=default_override_or(identity),
                           init=default_override_or(glorot_uniform()),
                           pad=default_override_or(False),
                           strides=1,
                           bias=default_override_or(True),
                           init_bias=default_override_or(0),
                           output_shape=None, 
                           name=''):
    '''
    ConvolutionTranspose2D(filter_shape, num_filters, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, output_shape=None, name='')

    Layer factory function to create a 2D convolution transpose layer with optional non-linearity.
    Same as `ConvolutionTranspose()` except that filter_shape is verified to be 2-dimensional.
    See `ConvolutionTranspose()` for extensive documentation.
    '''
    activation = get_default_override(ConvolutionTranspose2D, activation=activation)
    init       = get_default_override(ConvolutionTranspose2D, init=init)
    pad        = get_default_override(ConvolutionTranspose2D, pad=pad)
    bias       = get_default_override(ConvolutionTranspose2D, bias=bias)
    init_bias  = get_default_override(ConvolutionTranspose2D, init_bias=init_bias)
    output_shape = get_default_override(ConvolutionTranspose2D, output_shape=output_shape)
    if len(_as_tuple(filter_shape)) > 2: 
         raise ValueError('ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)')
    filter_shape = _pad_to_shape((0,0), filter_shape, 'filter_shape')
    return ConvolutionTranspose(filter_shape, num_filters, activation, init, pad, strides, True, bias, init_bias, output_shape, name=name)

# ConvolutionTranspose3D -- create a 3D convolution transpose layer with optional non-linearity
def ConvolutionTranspose3D(filter_shape,        # a 3D tuple, e.g., (3,3,3) 
                           num_filters,
                           activation=default_override_or(identity),
                           init=default_override_or(glorot_uniform()),
                           pad=default_override_or(False),
                           strides=1,
                           bias=default_override_or(True),
                           init_bias=default_override_or(0),
                           output_shape=None, 
                           name=''):
    '''
    ConvolutionTranspose3D(filter_shape, num_filters, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, output_shape=None, name='')

    Layer factory function to create a 3D convolution transpose layer with optional non-linearity.
    Same as `ConvolutionTranspose()` except that filter_shape is verified to be 3-dimensional.
    See `ConvolutionTranspose()` for extensive documentation.
    '''
    activation = get_default_override(ConvolutionTranspose3D, activation=activation)
    init       = get_default_override(ConvolutionTranspose3D, init=init)
    pad        = get_default_override(ConvolutionTranspose3D, pad=pad)
    bias       = get_default_override(ConvolutionTranspose3D, bias=bias)
    init_bias  = get_default_override(ConvolutionTranspose3D, init_bias=init_bias)
    output_shape = get_default_override(ConvolutionTranspose3D, output_shape=output_shape)
    if len(_as_tuple(filter_shape)) > 3: 
         raise ValueError('ConvolutionTranspose3D: filter_shape must be a scalar or a 3D tuple, e.g. 3 or (3,3,3)')
    filter_shape = _pad_to_shape((0,0,0), filter_shape, 'filter_shape')
    return ConvolutionTranspose(filter_shape, num_filters, activation, init, pad, strides, True, bias, init_bias, output_shape, name=name)

# TODO: add sequential mode like Convolution()
from cntk.cntk_py import PoolingType_Max, PoolingType_Average, NDShape
def _Pooling(op,           # PoolingType_Max or _Average
             filter_shape, # shape of receptive field, e.g. (3,3)
             sequential=False, # pooling in time if True (filter_shape[0] corresponds to dynamic axis)
             strides=1,
             pad=False,
             op_name=None,
             name=''):
    '''
    Shared part of the various pooling layers.
    Set the filter_shape to None to denote global pooling.
    '''

    if sequential:
        raise NotImplementedError("Pooling: sequential option not implemented yet")

    strides     = _pad_to_shape(filter_shape, strides, 'strides')
    pad         = _pad_to_shape(filter_shape, pad, 'pad')

    @BlockFunction(op_name, name)
    def pool(x):
        return pooling (x, op, filter_shape, strides=strides, auto_padding=pad)
    return pool


def MaxPooling(filter_shape,  # shape of receptive field, e.g. (3,3)
               strides=1,
               pad=default_override_or(False),
               name=''):
    '''
    MaxPooling(filter_shape, strides=1, pad=False, name='')

    Layer factory function to create a max-pooling layer.

    Like ``Convolution()``, ``MaxPooling()`` processes items arranged on an N-dimensional grid, such as an image.
    Typically, each item is a vector.
    For each item, max-pooling computes the element-wise maximum over a window ("receptive field") of items surrounding the item's position on the grid.

    The size (spatial extent) of the receptive field is given by ``filter_shape``.
    E.g. for 2D pooling, ``filter_shape`` should be a tuple of two integers, such as `(5,5)`.

    Example:
     >>> f = MaxPooling((3,3), strides=2)  # reduce dimensionality by 2, pooling over windows of 3x3
     >>> h = input((32,240,320))  # e.g. 32-dim feature map
     >>> hp = f(h)
     >>> hp.shape  # spatial dimension has been halved due to stride, and lost one due to 3x3 window without padding
         (32, 119, 159)

     >>> f = MaxPooling((2,2), strides=2)
     >>> f.update_signature((1,4,4))
     >>> im = np.array([[[3, 5, 2, 6], [4, 2, 8, 3], [1, 6, 4, 7], [7, 3, 5, 9]]])  # a 4x4 image (feature-map depth 1 for simplicity)
     >>> im
         array([[[3, 5, 2, 6],
                 [4, 2, 8, 3],
                 [1, 6, 4, 7],
                 [7, 3, 5, 9]]])
     >>> f([[im]])  # due to strides=2, this picks the max out of each 2x2 sub-block
         array([[[[ 5.,  8.],
                  [ 7.,  9.]]]], dtype=float32)

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride (increment when sliding over the input). Use a `tuple` to specify a per-axis value.
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the pooling operation will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      pooling will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the max-pooling operation to it
    '''
    pad = get_default_override(MaxPooling, pad=pad)
    return _Pooling(PoolingType_Max, filter_shape, strides=strides, pad=pad, op_name='MaxPooling', name=name)


def AveragePooling(filter_shape,  # shape of receptive field, e.g. (3,3)
                   strides=1,
                   pad=default_override_or(False),
                   name=''):
    '''
    AveragePooling(filter_shape, strides=1, pad=False, name='')

    Layer factory function to create an average-pooling layer.

    Like ``Convolution()``, ``AveragePooling()`` processes items arranged on an N-dimensional grid, such as an image.
    Typically, each item is a vector.
    For each item, average-pooling computes the element-wise mean over a window ("receptive field") of items surrounding the item's position on the grid.

    The size (spatial extent) of the receptive field is given by ``filter_shape``.
    E.g. for 2D pooling, ``filter_shape`` should be a tuple of two integers, such as `(5,5)`.

    Example:
     >>> f = AveragePooling((3,3), strides=2)  # reduce dimensionality by 2, pooling over windows of 3x3
     >>> h = input((32,240,320))  # e.g. 32-dim feature map
     >>> hp = f(h)
     >>> hp.shape  # spatial dimension has been halved due to stride, and lost one due to 3x3 window without padding
         (32, 119, 159)

     >>> f = AveragePooling((2,2), strides=2)
     >>> f.update_signature((1,4,4))
     >>> im = np.array([[[3, 5, 2, 6], [4, 2, 8, 3], [1, 6, 4, 7], [7, 3, 5, 9]]])  # a 4x4 image (feature-map depth 1 for simplicity)
     >>> im
         array([[[3, 5, 2, 6],
                 [4, 2, 8, 3],
                 [1, 6, 4, 7],
                 [7, 3, 5, 9]]])
     >>> f([[im]])  # due to strides=2, this computes the averages of each 2x2 sub-block
         array([[[[ 3.5 ,  4.75],
                  [ 4.25,  6.25]]]], dtype=float32)

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride (increment when sliding over the input). Use a `tuple` to specify a per-axis value.
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the pooling operation will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      pooling will be applied to all input positions, and positions outside the valid region will be excluded from the averaging.
      Use a `tuple` to specify a per-axis value.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the average-pooling operation to it
    '''
    pad = get_default_override(AveragePooling, pad=pad)
    return _Pooling(PoolingType_Average, filter_shape, strides=strides, pad=pad, op_name='AveragePooling', name=name)


def GlobalMaxPooling(name=''):
    '''
    Layer factory function to create a global max-pooling layer.

    The global max-pooling operation computes the element-wise maximum over all items on an N-dimensional grid, such as an image.

    This operation is the same as applying ``reduce_max()`` to all grid dimensions.

    Example:
     >>> f = GlobalMaxPooling()
     >>> f.update_signature((1,4,4))
     >>> im = np.array([[[3, 5, 2, 6], [4, 2, 8, 3], [1, 6, 4, 7], [7, 3, 5, 9]]])  # a 4x4 image (feature-map depth 1 for simplicity)
     >>> im
         array([[[3, 5, 2, 6],
                 [4, 2, 8, 3],
                 [1, 6, 4, 7],
                 [7, 3, 5, 9]]])
     >>> f([[im]])
         array([[[[ 9.]]]], dtype=float32)

    Args:
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''
    return _Pooling(PoolingType_Max, NDShape.unknown.dimensions(), pad=False, op_name='GlobalMaxPooling', name=name)


def GlobalAveragePooling(name=''):
    '''
    Layer factory function to create a global average-pooling layer.

    The global average-pooling operation computes the element-wise mean over all items on an N-dimensional grid, such as an image.

    This operation is the same as applying ``reduce_mean()`` to all grid dimensions.

    Example:
     >>> f = GlobalAveragePooling()
     >>> f.update_signature((1,4,4))
     >>> im = np.array([[[3, 5, 2, 6], [4, 2, 8, 3], [1, 6, 4, 7], [7, 3, 5, 9]]])  # a 4x4 image (feature-map depth 1 for simplicity)
     >>> im
         array([[[3, 5, 2, 6],
                 [4, 2, 8, 3],
                 [1, 6, 4, 7],
                 [7, 3, 5, 9]]])
     >>> f([[im]])
         array([[[[ 4.6875]]]], dtype=float32)

    Args:
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''
    return _Pooling(PoolingType_Average, NDShape.unknown.dimensions(), pad=False, op_name='GlobalAveragePooling', name=name)


# Create a max unpooling layer
# TODO: merge this. Test: Tests\EndToEndTests\CNTKv2Python\Examples\deconv_MNIST_test.py, Tests\EndToEndTests\Examples\Image\GettingStarted\07_Deconvolution
def MaxUnpooling(filter_shape,  # shape of receptive field, e.g. (3,3)
                 strides=1,
                 pad=False,
                 name=''):

    strides     = _pad_to_shape(filter_shape, strides, 'strides')
    pad         = _pad_to_shape(filter_shape, pad, 'pad')

    @BlockFunction('MaxUnpooling', name)
    def maxunpool(x, y):
        return unpooling (x, y, PoolingType_Max, filter_shape, strides=strides, auto_padding=pad)
    return maxunpool


# TODO: should the rate(s) be default_options?
def Dropout(dropout_rate=None, 
            keep_prob=None,
            seed = SentinelValueForAutoSelectRandomSeed,
            name=''):
    '''
    Layer factory function to create a drop-out layer.

    The dropout rate can be specified as the probability of *dropping* a value (``dropout_rate``).
    E.g. ``Dropout(0.3)`` means "drop 30% o the activation values."
    Alternatively, it can also be specified as the probability of *keeping* a value (``keep_prob``).

    Example:
     >>> f = Dropout(0.2)   # "drop 20% of activations"
     >>> h = input(3)
     >>> hd = f(h)

     >>> f = Dropout(keep_prob=0.8)   # "keep 80%"
     >>> h = input(3)
     >>> hd = f(h)

    Args:
     dropout_rate (float): probability of dropping out an element, mutually exclusive with ``keep_prob``
     keep_prob (float): probability of keeping an element, mutually exclusive with ``dropout_rate``
     seed (int): random seed.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''
    if dropout_rate is None and keep_prob is None:
        raise ValueError("Dense: either dropout_rate or keep_prob must be specified.")
    elif dropout_rate is not None and keep_prob is not None:
        raise ValueError("Dense: dropout_rate and keep_prob cannot be specified at the same time.")
    elif keep_prob is not None:
        dropout_rate = 1-keep_prob
    @BlockFunction('Dropout', name)
    def dropout_f(x):
        return dropout(x, dropout_rate=dropout_rate, seed=seed)
    return dropout_f


def Activation(activation=default_override_or(identity), name=''): 
    '''
    Activation(activation=identity, name='')

    Layer factory function to create an activation layer.
    Activation functions can be used directly in CNTK, so there is no difference between
    ``y = relu(x)`` and ``y = Activation(relu)(x)``.
    This layer is useful if one wants to configure the activation function
    with ``default_options``, or when its invocation should be named.

    Example:
     >>> model = Dense(500) >> Activation(C.relu) >> Dense(10)
     >>> # is the same as
     >>> model = Dense(500) >> C.relu >> Dense(10)
     >>> # and also the same as
     >>> model = Dense(500, activation=C.relu) >> Dense(10)

    Args:
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): function to apply at the end, e.g. `relu`
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''
    activation = get_default_override(Activation, activation=activation)
    @BlockFunction('Activation', name)
    def activation_f(x):
        return activation(x) 
    return activation_f


# TODO: map_rank is broken. We should specify the #slowest-changing axes. E.g. 1 would work for images and vectors. Requires C++ change.
def BatchNormalization(map_rank=default_override_or(None),  # if given then normalize only over this many dimensions. E.g. pass 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=default_override_or(5000), blend_time_constant=0,
                       epsilon=default_override_or(0.00001), use_cntk_engine=default_override_or(False),
                       name=''):
    '''
    BatchNormalization(map_rank=None, init_scale=1, normalization_time_constant=5000, blend_time_constant=0, epsilon=0.00001, use_cntk_engine=False, name='')

    Layer factory function to create a batch-normalization layer.

    Batch normalization applies this formula to every input element (element-wise):
    ``y = (x - batch_mean) / (batch_stddev + epsilon) * scale + bias``
    where ``batch_mean`` and ``batch_stddev`` are estimated on the minibatch and ``scale`` and ``bias`` are learned parameters.
    TODO: add paper reference

    During operation, this layer also estimates an aggregate running mean and standard deviation for use in inference.

    A ``BatchNormalization`` layer instance owns its learnable parameter tensors and exposes them as attributes ``.scale`` and ``.bias``.
    The aggregate estimates are exposed as attributes ``aggregate_mean``, ``aggregate_variance``, and ``aggregate_count``.

    Example:
     >>> # BatchNorm on an image with spatial pooling
     >>> f = BatchNormalization(map_rank=1)
     >>> f.update_signature((3,480,640))
     >>> f.bias.shape, f.scale.shape  # due to spatial pooling (map_rank=1), there are only 3 biases and scales, shared across all pixel positions
         ((3,), (3,))

    Args:
     map_rank (1 or ``None``): passing 1 means spatially-pooled batch-normalization, where normalization values will be tied across all pixel positions; while ``None``
      will normalize all elements of the input tensor independently
     init_scale (float, default 1): initial value for the ``scale`` parameter
     normalization_time_constant (int, default 5000): time constant for smoothing the batch statistics in order to compute aggregate estimates for inference.
     epsilon (float, default 0.00001): epsilon added to the variance to avoid division by 0
     use_cntk_engine (bool, default ``False``): if ``True`` then use CNTK's own engine instead of NVidia's.
     name (str, optional): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
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
    run_mean     = Constant(0, shape=norm_shape, name='aggregate_mean')  # note: these are not really constants; they are updated differently
    run_variance = Constant(0, shape=norm_shape, name='aggregate_variance')
    run_count    = Constant(0, shape=(),         name='aggregate_count')

    # expression
    @BlockFunction('BatchNormalization', name)
    def batch_normalize(x):
        return batch_normalization(x, scale, bias, run_mean, run_variance, running_count=run_count,
                                   spatial=map_rank == 1, normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, epsilon=epsilon,
                                   use_cudnn_engine=not use_cntk_engine)

    return batch_normalize

def LayerNormalization(initial_scale=1, initial_bias=0, epsilon=default_override_or(0.00001), name=''):    
    '''
    LayerNormalization(initial_scale=1, initial_bias=0, epsilon=0.00001, name='')

    Layer factory function to create a function that implements layer normalization.

    Layer normalization applies this formula to every input element (element-wise):
    ``y = (x - mean(x)) / (stddev(x) + epsilon) * scale + bias``
    where ``scale`` and ``bias`` are learned scalar parameters.
    TODO: add paper reference

    Example:
     >>> f = LayerNormalization(initial_scale=2, initial_bias=1)
     >>> f.update_signature(4)
     >>> f([np.array([4,0,0,4])])  # result has mean 1 and standard deviation 2, reflecting the initial values for scale and bias
         array([[ 2.99999, -0.99999, -0.99999,  2.99999]], dtype=float32)

    Args:
     initial_scale (float, default 1): initial value for the ``scale`` parameter
     initial_bias (float, default 0): initial value for the ``bias`` parameter
     epsilon (float, default 0.00001): epsilon added to the standard deviation to avoid division by 0
     name (str, optional): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it
    '''
    epsilon = get_default_override(LayerNormalization, epsilon=epsilon)

    # parameters bound to this Function
    scale = Parameter((), init=initial_scale, name='scale')  # TODO: if this gets usage then offer a Softplus version like Stabilizer() for stability?
    bias  = Parameter((), init=initial_bias,  name='bias')

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
    This can be used to access an intermediate value flowing through computation.

    Args:
     name (str): the name of the function instance in the network

    Example:
     >>> model = Dense(500) >> Label('hidden') >> Dense(10)
     >>> model.update_signature(10)
     >>> intermediate_val = model.hidden
     >>> intermediate_val.shape
         (500,)

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and returns it with the desired name attached
    '''
    @Function  # note: cannot be a BlockFunction since that would hide the label
    def label(x):
        return alias(x, name=name)
    return label
