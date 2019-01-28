# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
CNTK core operators. Calling these operators creates nodes in the CNTK computational graph.
"""


from __future__ import division
from __future__ import print_function
import numpy as np
import numbers
from . import sequence
from .functions import ModelFormat, CloneMethod, Function, BlockFunction, load_model, register_native_user_function, native_user_function
from cntk.internal import sanitize_input, sanitize_shape, sanitize_axis, sanitize_dynamic_axes, sanitize_axis_list, sanitize_multi_axis_reduction_list, typemap, sanitize_pooling_args, sanitize_convolution_args, sanitize_permutation, sanitize_dtype_cntk
from cntk.internal.utils import get_data_type
from ..axis import Axis
from .. import cntk_py
from ..cntk_py import sentinel_value_for_auto_select_random_seed as SentinelValueForAutoSelectRandomSeed
from ..default_options import get_default_override, default_override_or

TIMES_NO_INFERRED_INPUT_RANK                            = cntk_py.TimesNoInferredInputRank
TIMES_REDUCE_SEQUENCE_AXIS_WITHOUT_INFERRED_INPUT_RANK  = cntk_py.TimesReduceSequenceAxisWithoutInferredInputRank

CONSTANT_PAD = cntk_py.PaddingMode_CONSTANTPAD
REFLECT_PAD = cntk_py.PaddingMode_REFLECTPAD
SYMMETRIC_PAD = cntk_py.PaddingMode_SYMMETRICPAD

@typemap
def combine(*operands, **kw_name):
    '''
     Create a new Function instance which just combines the outputs of the specified list of
     'operands' Functions such that the 'Outputs' of the new 'Function' are union of the
     'Outputs' of each of the specified 'operands' Functions. E.g., when creating a classification
     model, typically the CrossEntropy loss Function and the ClassificationError Function comprise
     the two roots of the computation graph which can be combined to create a single Function
     with 2 outputs; viz. CrossEntropy loss and ClassificationError output.

    Example:
        >>> in1 = C.input_variable((4,))
        >>> in2 = C.input_variable((4,))

        >>> in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
        >>> in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)

        >>> plus_operation = in1 + in2
        >>> minus_operation = in1 - in2

        >>> forward = C.combine([plus_operation, minus_operation]).eval({in1: in1_data, in2: in2_data})
        >>> len(forward)
        2
        >>> list(forward.values()) # doctest: +SKIP
        [array([[[ 1., -3.,  6.,  2.]]], dtype=float32),
         array([[[ 1.,  7.,  0.,  6.]]], dtype=float32)]
        >>> x = C.input_variable((4,))
        >>> _ = C.combine(x, x)
        >>> _ = C.combine([x, x])
        >>> _ = C.combine((x, x))
        >>> _ = C.combine(C.combine(x, x), x)

    Args:
        operands (list): list of functions or their variables to combine
        name (str, optional): the name of the Combine Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': (name))(**kw_name) # Python 2.7 does not allow (*inputs, name='')

    from cntk.cntk_py import combine
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]
    if isinstance(operands, tuple):
        operands = list(operands)
    operands_unfold = []
    for o in operands:
        if hasattr(o, 'outputs') and len(o.outputs) > 1:
            operands_unfold += o.outputs
        else:
            operands_unfold += [o]
    return combine(operands_unfold, name)

@typemap
def mean(*operands, **kw_name):
    '''
     Create a new Function instance that computes element-wise mean of input tensors.

    Example:
        >>> in1 = C.input_variable((4,))
        >>> in2 = C.input_variable((4,))
        >>> model = C.mean([in1, in2])
        >>> in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
        >>> in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)
        >>> model.eval({in1: in1_data, in2: in2_data})
        array([[ 0.5,  3.5,  0. ,  3. ]], dtype=float32)

    Args:
        operands (list): list of functions
        name (str, optional): the name of the mean Function in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': (name))(**kw_name) # Python 2.7 does not allow (*inputs, name='')

    from cntk.cntk_py import mean
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]
    if isinstance(operands, tuple):
        operands = list(operands)
    operands_unfold = []
    for o in operands:
        if hasattr(o, 'outputs') and len(o.outputs) > 1:
            operands_unfold += o.outputs
        else:
            operands_unfold += [o]
    return mean(operands_unfold, name)

@typemap
def sum(*operands, **kw_name):
    '''
     Create a new Function instance that computes element-wise sum of input tensors.

    Example:
        >>> in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
        >>> in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)
        >>> in1 = C.input_variable(np.shape(in1_data))
        >>> in2 = C.input_variable(np.shape(in2_data))
        >>> C.sum([in1, in2]).eval({in1: in1_data, in2: in2_data})
        array([[[ 1.,  7.,  0.,  6.]]], dtype=float32)

    Args:
        operands (list): list of functions
        name (str, optional): the name of the sum Function in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    name = (lambda name='': (name))(**kw_name) # Python 2.7 does not allow (*inputs, name='')

    from cntk.cntk_py import sum
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]
    if isinstance(operands, tuple):
        operands = list(operands)
    operands_unfold = []
    for o in operands:
        if hasattr(o, 'outputs') and len(o.outputs) > 1:
            operands_unfold += o.outputs
        else:
            operands_unfold += [o]
    return sum(operands_unfold, name)

@typemap
def as_block(composite, block_arguments_map, block_op_name, block_instance_name=''):
    '''
     Create a new block Function instance which just encapsulates the specified composite Function
     to create a new Function that appears to be a primitive. All the arguments of the composite
     being encapsulated must be Placeholder variables.
     The purpose of block Functions is to enable creation of hierarchical Function graphs
     where details of implementing certain building block operations can be encapsulated away
     such that the actual structure of the block's implementation is not inlined into
     the parent graph where the block is used, and instead the block just appears as an opaque
     primitive. Users still have the ability to peek at the underlying Function graph that implements
     the actual block Function.

    Args:
        composite: The composite Function that the block encapsulates
        block_arguments_map: A list of tuples, mapping from block's underlying composite's arguments to
         actual variables they are connected to
        block_op_name: Name of the op that the block represents
        block_instance_name (str, optional): the name of the block Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import as_block
    return as_block(composite, block_arguments_map, block_op_name, block_instance_name)

@typemap
def as_composite(root_function, name=''):
    '''
     Creates a composite Function that has the specified root_function as its root.
     The composite denotes a higher-level Function encapsulating the entire graph
     of Functions underlying the specified rootFunction.

    Args:
        root_function: Root Function, the graph underlying which, the newly created composite encapsulates
        name (str, optional): the name of the Alias Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import as_composite
    return as_composite(root_function, name)

@typemap
def alias(x, name=''):
    '''
     Create a new Function instance which just aliases the specified 'x' Function/Variable
     such that the 'Output' of the new 'Function' is same as the 'Output' of the specified
     'x' Function/Variable, and has the newly specified name.
     The purpose of this operator is to create a new distinct reference to a symbolic
     computation which is different from the original Function/Variable that it aliases and can
     be used for e.g. to substitute a specific instance of the aliased Function/Variable in the
     computation graph instead of substituting all usages of the aliased Function/Variable.

    Args:
        operand: The Function/Variable to alias
        name (str, optional): the name of the Alias Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import alias
    x = sanitize_input(x)
    return alias(x, name)

@typemap
def reconcile_dynamic_axes(x, dynamic_axes_as, name=''):
    '''
     Create a new Function instance which reconciles the dynamic axes of the
     specified tensor operands. The output of the returned Function has the sample
     layout of the 'x' operand and the dynamic axes of the 'dynamic_axes_as' operand.
     This operator also performs a runtime check to ensure that the dynamic axes layouts
     of the 2 operands indeed match.

    Args:
        x: The Function/Variable, whose dynamic axes are to be reconciled
        dynamic_axes_as: The Function/Variable, to whose dynamic axes the
            operand 'x''s dynamic axes are reconciled to.
        name (str, optional): the name of the reconcile_dynamic_axes Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reconcile_dynamic_axes
    x = sanitize_input(x)
    dynamic_axes_as = sanitize_input(dynamic_axes_as)
    return reconcile_dynamic_axes(x, dynamic_axes_as, name)

@typemap
def labels_to_graph(labels, name=''):
    '''
    Conversion node from labels to graph. Typically used as an input to ForwardBackward node.
    This node's objective is to transform input labels into a graph representing exact forward-backward criterion.

    Example:
        >>> num_classes = 2
        >>> labels = C.input_variable((num_classes))
        >>> graph = C.labels_to_graph(labels)

    Args:
        labels: input training labels
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import labels_to_graph
    dtype = get_data_type(labels)
    labels = sanitize_input(labels, dtype)
    return labels_to_graph(labels, name)

@typemap
def forward_backward(graph, features, blankTokenId, delayConstraint=-1, name=''):
    '''
    Criterion node for training methods that rely on forward-backward Viterbi-like passes,
    e.g. Connectionist Temporal Classification (CTC) training
    The node takes as the input the graph of labels, produced by the labels_to_graph
    operation that determines the exact forward/backward procedure.

    Example:
        graph = cntk.labels_to_graph(labels)
        networkOut = model(features)
        fb = C.forward_backward(graph, networkOut, 132)

    This op requires that both graph and features have the same dynamic sequence axis (i.e. same sequence length).

    Labels feed into cntk.labels_to_graph and hence to forward_backward are represented as 1-hot vectors
    for each frame. The 1-hot vectors may have either value 1 or 2 at the position of the phone
    corresponding to the frame, where the value 1 means the frame is within phone boundary (i.e. not the
    first frame of the phone) and 2 means the frame is the phone boundary (i.e. is the first frame of the phone).
    If you are using HTKMLFDeserializer, this encoding will be automatically done.

    Example:
        labels = [[0, 2, 0, 0, 0],  # first frame of label_id = 1
                  [0, 1, 0, 0, 0],  # second frame of label_id = 1
                  [0, 0, 2, 0, 0],  # first frame of label_id = 2
                  [0, 2, 0, 0, 0],] # first frame of label_id = 1

    For cases where labels are not frame-aligned (i.e. sequence length of labels is shorter than the
    sequence length of features), you can pad the labels by duplicating the last label and setting the value as 1
    until the sequence length is equal to features.

    Alternatively, you can also generate the labels to have uniform (equal) distribution of the labels across
    the feature frames (keeping in mind to set the value in the one hot encoding appropriately (1 or 2 depending
    on whether label is the first frame or not).

    Example:
        # Padding labels by duplicating the last label

        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer(pos_label=2).fit(range(6))
        labels = lb.transform([0, 2, 0, 1, 3, 4]) # blank is the label 5

        # labels = [[2,0,0,...,0,0], [0,0,2,...,0,0], ..., [0,0,0,...,2,0]]

        # Retrieve the input's sequence length
        sequence_dim = input.shape[-2]
        expanded_labels = np.zeros((sequence_dim, labels.shape[-1]))

        # We first copy the original one-hot labels
        expanded_labels[:len(labels)] = labels

        # Then, we replicate the last label as one-hot-1 encoded
        expanded_labels[len(labels):, labels[-1].argmax()] = 1

        # expanded_labels = [[2,0,0,...,0,0], [0,0,2,...,0,0], ...,
        #                     [0,0,0,...,2,0], [0,0,0,...,1,0], ...,
        #                     [0,0,0,...,1,0]]

        # We can define the model and the variables
        input_var = sequence.input_variable((input.shape[-1]), name='input')
        labels_var = sequence.input_variable((6), name='label')

        # The model should be defined here
        # model = ...

        # Now, we can use the forward-backward algorithm
        labels_graph = cntk.labels_to_graph(labels_var)
        network_out = model(input_var)
        fb = forward_backward(labels_graph, network_out, 5)
        fb.eval({'input': input.astype(np.float32), 'label': expanded_labels.astype(np.float32)})

    Args:
        graph: labels graph
        features: network output
        blankTokenId: id of the CTC blank label
        delayConstraint: label output delay constraint introduced during training that allows to have shorter
         delay during inference. This is using the original time information to enforce that CTC tokens only get
         aligned within a time margin. Setting this parameter smaller will result in shorted delay between label
         output during decoding, yet may hurt accuracy. delayConstraint=-1 means no constraint
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import forward_backward
    dtype = get_data_type(features, graph)
    features = sanitize_input(features, dtype)
    graph = sanitize_input(graph, dtype)
    return forward_backward(graph, features, blankTokenId, delayConstraint, name)

##########################################################################
# convolution ops
##########################################################################

@typemap
def convolution(convolution_map, operand, strides=(1,), sharing=[True],
                auto_padding=[True], sequential=False, dilation=(1,), reduction_rank=1, groups=1, max_temp_mem_size_in_samples=0, name=''):
    '''
    Computes the convolution of ``convolution_map`` (typically a tensor of learnable parameters) with
    ``operand`` (commonly an image or output of a previous convolution/pooling operation).
    This operation is used in image and language processing applications. It supports arbitrary
    dimensions, strides, sharing, and padding.

    This function operates on input tensors with dimensions :math:`[C \\times M_1 \\times M_2 \\times \\ldots \\times M_n]`. This can be understood as a rank-n
    object, where each entry consists of a :math:`C`-dimensional vector. For example, an RGB image would have dimensions
    :math:`[3 \\times W \\times H]`, i.e. a :math:`[W \\times H]`-sized structure, where each entry (pixel) consists of a 3-tuple.

    `convolution` convolves the input ``operand`` with a :math:`n+2` rank tensor of (typically learnable) filters called
    ``convolution_map`` of shape :math:`[O \\times I \\times m_1 \\times m_2 \\times \\ldots \\times m_n ]` (typically :math:`m_i \\ll M_i`).
    The first dimension, :math:`O`, is the number of convolution filters (i.e. the number of
    channels in the output). The second dimension, :math:`I`, must match the number of channels in the input, which can be ignored if `reduction_rank` is `0`.
    The last n dimensions are the spatial extent of the filter. I.e. for each output position, a vector of
    dimension :math:`O` is computed. Hence, the total number of filter parameters is :math:`O \\times I \\times m_1 \\times m_2 \\times \\ldots \\times m_n`


    Example:
        >>> img = np.reshape(np.arange(25.0, dtype = np.float32), (1, 5, 5))
        >>> x = C.input_variable(img.shape)
        >>> filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
        >>> kernel = C.constant(value = filter)
        >>> np.round(C.convolution(kernel, x, auto_padding = [False]).eval({x: [img]}),5)
        array([[[[  6.,   8.,  10.,  12.],
                  [ 16.,  18.,  20.,  22.],
                  [ 26.,  28.,  30.,  32.],
                  [ 36.,  38.,  40.,  42.]]]], dtype=float32)

    Args:
        convolution_map: convolution filter weights, stored as a tensor of dimensions :math:`[O \\times I \\times m_1 \\times m_2 \\times \\ldots \\times m_n]`,
         where :math:`[m_1 \\times m_2 \\times \\ldots \\times m_n]` must be the kernel dimensions (spatial extent of the filter).
        operand: convolution input. A tensor with dimensions :math:`[I \\times M_1 \\times M_2 \\times \\ldots \\times M_n]`.
        strides (tuple, optional): stride dimensions. If strides[i] > 1 then only pixel positions that are multiples of strides[i] are computed.
         For example, a stride of 2 will lead to a halving of that dimension. The first stride dimension that lines up with the number
         of input channels can be set to any non-zero value.
        sharing (bool): sharing flags for each input dimension
        auto_padding (bool): flags for each input dimension whether it should be padded automatically (that is,
         symmetrically) or not padded at all. Padding means that the convolution kernel is applied to all pixel positions, where all
         pixels outside the area are assumed zero ("padded with zeroes"). Without padding, the kernels are only shifted over
         positions where all inputs to the kernel still fall inside the area. In this case, the output dimension will be less than
         the input dimension. The last value that lines up with the number of input channels must be false.
        dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
        reduction_rank (`int`, default 1): must be 0 or 1, 0 mean no depth or channel dimension in the input and 1 mean the input has channel or depth dimension.
        groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1, 
         which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
         divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
         Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0. 
        sequential (bool, default False): flag if convolve over sequential axis. 
        max_temp_mem_size_in_samples (int): maximum amount of auxiliary memory (in samples) that should be reserved to perform convolution
         operations. Some convolution engines (e.g. cuDNN and GEMM-based engines) can benefit from using workspace as it may improve
         performance. However, sometimes this may lead to higher memory utilization. Default is 0 which means the same as the input
         samples.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import convolution
    operand = sanitize_input(operand)
    strides, sharing, auto_padding = sanitize_convolution_args(strides, sharing, auto_padding)
    dilation = sanitize_shape(dilation)
    return convolution(convolution_map, operand, strides, sharing, auto_padding, dilation,
                       reduction_rank, groups, max_temp_mem_size_in_samples, sequential, name)

@typemap
def convolution_transpose(convolution_map, operand, strides=(1,), sharing=[True],
                          auto_padding=[True], output_shape=None, dilation=(1,), reduction_rank=1, max_temp_mem_size_in_samples=0, name=''):
    '''
    Computes the transposed convolution of ``convolution_map`` (typically a tensor of learnable parameters) with
    ``operand`` (commonly an image or output of a previous convolution/pooling operation).
    This is also known as ``fractionally strided convolutional layers``, or, ``deconvolution``.
    This operation is used in image and language processing applications. It supports arbitrary
    dimensions, strides, sharing, and padding.

    This function operates on input tensors with dimensions :math:`[C \\times M_1 \\times M_2 \\times \\ldots \\times M_n]`. This can be understood as a rank-n
    object, where each entry consists of a :math:`C`-dimensional vector. For example, an RGB image would have dimensions
    :math:`[3 \\times W \\times H]`, i.e. a :math:`[W \\times H]`-sized structure, where each entry (pixel) consists of a 3-tuple.

    `convolution_transpose` convolves the input ``operand`` with a :math:`n+2` rank tensor of (typically learnable) filters called
    ``convolution_map`` of shape :math:`[I \\times O \\times m_1 \\times m_2 \\times \\ldots \\times m_n ]` (typically :math:`m_i \\ll M_i`).
    The first dimension, :math:`I`, must match the number of channels in the input. The second dimension, :math:`O`, is the number of convolution filters (i.e. the number of
    channels in the output).
    The last n dimensions are the spatial extent of the filter. I.e. for each output position, a vector of
    dimension :math:`O` is computed. Hence, the total number of filter parameters is :math:`I \\times O \\times m_1 \\times m_2 \\times \\ldots \\times m_n`


    Example:
        >>> img = np.reshape(np.arange(9.0, dtype = np.float32), (1, 3, 3))
        >>> x = C.input_variable(img.shape)
        >>> filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
        >>> kernel = C.constant(value = filter)
        >>> np.round(C.convolution_transpose(kernel, x, auto_padding = [False]).eval({x: [img]}),5)
        array([[[[  0.,   2.,   3.,  -2.],
                  [  6.,   4.,   6.,  -1.],
                  [  9.,  10.,  12.,   2.],
                  [ -6.,   5.,   6.,  16.]]]], dtype=float32)

    Args:
        convolution_map: convolution filter weights, stored as a tensor of dimensions :math:`[I \\times O \\times m_1 \\times m_2 \\times \\ldots \\times m_n]`,
         where :math:`[m_1 \\times m_2 \\times \\ldots \\times m_n]` must be the kernel dimensions (spatial extent of the filter).
        operand: convolution input. A tensor with dimensions :math:`[I \\times M_1 \\times M_2 \\times \\ldots \\times M_n]`.
        strides (tuple, optional): stride dimensions. If strides[i] > 1 then only pixel positions that are multiples of strides[i] are computed.
         For example, a stride of 2 will lead to a halving of that dimension. The first stride dimension that lines up with the number
         of input channels can be set to any non-zero value.
        sharing (bool): sharing flags for each input dimension
        auto_padding (bool): flags for each input dimension whether it should be padded automatically (that is,
         symmetrically) or not padded at all. Padding means that the convolution kernel is applied to all pixel positions, where all
         pixels outside the area are assumed zero ("padded with zeroes"). Without padding, the kernels are only shifted over
         positions where all inputs to the kernel still fall inside the area. In this case, the output dimension will be less than
         the input dimension. The last value that lines up with the number of input channels must be false.
        output_shape: user expected output shape after convolution transpose.
        dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
        reduction_rank (`int`, default 1): must be 0 or 1, 0 mean no depth or channel dimension in the input and 1 mean the input has channel or depth dimension.
        max_temp_mem_size_in_samples (int): maximum amount of auxiliary memory (in samples) that should be reserved to perform convolution
         operations. Some convolution engines (e.g. cuDNN and GEMM-based engines) can benefit from using workspace as it may improve
         performance. However, sometimes this may lead to higher memory utilization. Default is 0 which means the same as the input
         samples.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import convolution_transpose
    operand = sanitize_input(operand)
    strides, sharing, auto_padding = sanitize_convolution_args(strides, sharing, auto_padding)
    if output_shape is None:
        output_shape = (0,)
    output_shape = sanitize_shape(output_shape)
    dilation = sanitize_shape(dilation)
    return convolution_transpose(convolution_map, operand, strides, sharing, auto_padding,
                                 output_shape, dilation, reduction_rank, max_temp_mem_size_in_samples, name)

from cntk.cntk_py import PoolingType_Max, PoolingType_Average
MAX_POOLING = PoolingType_Max
'''int: constant used to specify maximum pooling'''

AVG_POOLING = PoolingType_Average
'''int: constant used to specify average pooling'''

@typemap
def roipooling(operand, rois, pooling_type, roi_output_shape, spatial_scale, name=''):
    '''
    The ROI (Region of Interest) pooling operation pools over sub-regions of an input volume and produces
    a fixed sized output volume regardless of the ROI size. It is used for example for object detection.

    Each input image has a fixed number of regions of interest, which are specified as bounding boxes (x, y, w, h)
    that are relative to the image size [W x H]. This operation can be used as a replacement for the final
    pooling layer of an image classification network (as presented in Fast R-CNN and others).

    .. versionchanged:: 2.1
      The signature was updated to match the Caffe implementation:
      the parameters `pooling_type` and `spatial_scale` were added, and
      the coordinates for the parameters `rois` are now absolute to the original image size.

    Args:
        operand: a convolutional feature map as the input volume ([W x H x C x N]).
        pooling_type: only :const:`~cntk.ops.MAX_POOLING`
        rois: the coordinates of the ROIs per image ([4 x roisPerImage x N]), each ROI is (x1, y1, x2, y2) absolute to original image size.
        roi_output_shape: dimensions (width x height) of the ROI pooling output shape
        spatial_scale: the scale of operand from the original image size.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import roipooling
    if pooling_type != MAX_POOLING:
        raise ValueError('Unsupported pooling type, ROIPooling support only MAX pooling.')

    operand = sanitize_input(operand)
    rois = sanitize_input(rois)
    roi_output_shape = sanitize_shape(roi_output_shape)
    return roipooling(operand, rois, pooling_type, roi_output_shape, spatial_scale, name)

@typemap
def pooling(operand, pooling_type, pooling_window_shape, strides=(1,), auto_padding=[False],
            ceil_out_dim=False, include_pad=False, name=''):
    '''
    The pooling operations compute a new tensor by selecting the maximum or average value in the pooling input.
    In the case of average pooling with padding, the average is only over the valid region.

    N-dimensional pooling allows to create max or average pooling of any dimensions, stride or padding.

    Example:
        >>> img = np.reshape(np.arange(16, dtype = np.float32), [1, 4, 4])
        >>> x = C.input_variable(img.shape)
        >>> C.pooling(x, C.AVG_POOLING, (2,2), (2,2)).eval({x : [img]})
        array([[[[  2.5,   4.5],
                  [ 10.5,  12.5]]]], dtype=float32)
        >>> C.pooling(x, C.MAX_POOLING, (2,2), (2,2)).eval({x : [img]})
        array([[[[  5.,   7.],
                  [ 13.,  15.]]]], dtype=float32)

    Args:
        operand: pooling input
        pooling_type: one of :const:`~cntk.ops.MAX_POOLING` or :const:`~cntk.ops.AVG_POOLING`
        pooling_window_shape: dimensions of the pooling window
        strides (default 1): strides.
        auto_padding (default [False,]): automatic padding flags for each input dimension.
        ceil_out_dim (default False): ceiling while computing output size
        include_pad(default False): include pad while average pooling
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import pooling
    operand = sanitize_input(operand)
    pooling_window_shape, strides, auto_padding = sanitize_pooling_args(pooling_window_shape, strides, auto_padding)
    return pooling(operand, pooling_type, pooling_window_shape, strides, auto_padding,
                   ceil_out_dim, include_pad, name)


MAX_UNPOOLING = PoolingType_Max
'''int: constant used to specify maximum unpooling'''

@typemap
def unpooling(operand, pooling_input, unpooling_type, unpooling_window_shape, strides=(1,), auto_padding=[False],
              name=''):
    '''
    Unpools the ``operand`` using information from ``pooling_input``. Unpooling mirrors the operations
    performed by pooling and depends on the values provided to the corresponding pooling operation. The output
    should have the same shape as pooling_input. Pooling the result of an unpooling operation should
    give back the original input.

    Example:
        >>> img = np.reshape(np.arange(16, dtype = np.float32), [1, 4, 4])
        >>> x = C.input_variable(img.shape)
        >>> y = C.pooling(x, C.MAX_POOLING, (2,2), (2,2))
        >>> C.unpooling(y, x, C.MAX_UNPOOLING, (2,2), (2,2)).eval({x : [img]})
        array([[[[  0.,   0.,   0.,   0.],
                  [  0.,   5.,   0.,   7.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  13.,   0.,  15.]]]], dtype=float32)

    Args:
        operand: unpooling input
        pooling_input: input to the corresponding pooling operation
        unpooling_type: only :const:`~cntk.ops.MAX_UNPOOLING` is supported now
        unpooling_window_shape: dimensions of the unpooling window
        strides (default 1): strides.
        auto_padding: automatic padding flags for each input dimension.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import unpooling
    operand = sanitize_input(operand)
    pooling_input = sanitize_input(pooling_input)
    unpooling_window_shape, strides, auto_padding = sanitize_pooling_args(unpooling_window_shape, strides, auto_padding)
    return unpooling(operand, pooling_input, unpooling_type,
                     unpooling_window_shape, strides, auto_padding, name)


@typemap
def batch_normalization(operand, scale, bias, running_mean, running_inv_std, spatial,
                        normalization_time_constant=5000, blend_time_constant=0,
                        epsilon=0.00001, use_cudnn_engine=False, disable_regularization=False, name='', running_count=None):
    # TODO: running_count should be right after running_inv_std; no need for upwards compat
    '''
    Normalizes layer outputs for every minibatch for each output (feature) independently
    and applies affine transformation to preserve representation of the layer.

    Args:
        operand: input of the batch normalization operation
        scale: parameter tensor that holds the learned componentwise-scaling factors
        bias: parameter tensor that holds the learned bias. ``scale`` and ``bias`` must have the same
         dimensions which must be equal to the input dimensions in case of ``spatial`` = False or
         number of output convolution feature maps in case of ``spatial`` = True
        running_mean: running mean which is used during evaluation phase and might be used during
         training as well. You must pass a constant tensor with initial value 0 and the same dimensions
         as ``scale`` and ``bias``
        running_inv_std: running variance. Represented as ``running_mean``
        running_count: Denotes the total number of samples that have been used so far to compute
         the ``running_mean`` and ``running_inv_std`` parameters. You must pass a scalar (either rank-0 ``constant(val)``).
        spatial(bool): flag that indicates whether to compute mean/var for each feature in a minibatch
         independently or, in case of convolutional layers, per future map
        normalization_time_constant(float, default 5000): time constant for computing running average of
         mean and variance as a low-pass filtered version of the batch statistics.
        blend_time_constant(float, default 0): constant for smoothing batch estimates with the running
         statistics
        epsilon: conditioner constant added to the variance when computing the inverse standard deviation
        use_cudnn_engine(bool, default False):
        name (str, optional): the name of the Function instance in the network
        disable_regularization(bool, default False): turn off regularization in batch normalization
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    if running_count is None:
        running_count = constant(0)
        import warnings
        warnings.warn("batch_normalization requires an additional "
            "'running_count' parameter, which can be "
            "instantiated as 'constant(0)'", Warning)

    from cntk.cntk_py import batch_normalization
    operand = sanitize_input(operand)
    return batch_normalization(operand, scale, bias, running_mean, running_inv_std, running_count, spatial,
                               normalization_time_constant, blend_time_constant,
                               epsilon, use_cudnn_engine, disable_regularization, name)

@typemap
def local_response_normalization(operand, depth_radius, bias, alpha, beta, name=''):
    '''
    Local Response Normalization layer. See Section 3.3 of the paper:

    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    The mathematical equation is:

    ``b_{x,y}^i=a_{x,y}^i/(bias+\\alpha\\sum_{j=max(0,i-depth_radius)}^{min(N-1, i+depth_radius)}(a_{x,y}^j)^2)^\\beta``

    where a_{x,y}^i is the activity of a neuron computed by applying kernel i at position (x,y)
    N is the total number of kernels, depth_radius is half normalization width.

    Args:
        operand: input of the Local Response Normalization.
        depth_radius (int): the radius on the channel dimension to apply the normalization.
        bias (double): a bias term to avoid divide by zero.
        alpha (double): the alpha term of the above equation.
        beta (double): the beta term of the above equation.
        name (str, optional): the name of the Function instance in the network.
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import local_response_normalization
    operand = sanitize_input(operand)
    return local_response_normalization(operand, depth_radius, bias, alpha, beta, name)

##########################################################################
# comparison ops
##########################################################################

@typemap
def less(left, right, name=''):
    '''
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Example:
       >>> C.less([41., 42., 43.], [42., 42., 42.]).eval()
       array([ 1.,  0.,  0.], dtype=float32)

       >>> C.less([-1,0,1], [0]).eval()
       array([ 1.,  0.,  0.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import less
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return less(left, right, name)


@typemap
def equal(left, right, name=''):
    '''
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.

    Example:
        >>> C.equal([41., 42., 43.], [42., 42., 42.]).eval()
        array([ 0.,  1.,  0.], dtype=float32)

        >>> C.equal([-1,0,1], [1]).eval()
        array([ 0.,  0.,  1.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import equal
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return equal(left, right, name)


@typemap
def greater(left, right, name=''):
    '''
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.

    Example:
        >>> C.greater([41., 42., 43.], [42., 42., 42.]).eval()
        array([ 0.,  0.,  1.], dtype=float32)

        >>> C.greater([-1,0,1], [0]).eval()
        array([ 0.,  0.,  1.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import greater
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return greater(left, right, name)


@typemap
def greater_equal(left, right, name=''):
    '''
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Example:
        >>> C.greater_equal([41., 42., 43.], [42., 42., 42.]).eval()
        array([ 0.,  1.,  1.], dtype=float32)

        >>> C.greater_equal([-1,0,1], [0]).eval()
        array([ 0.,  1.,  1.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import greater_equal
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return greater_equal(left, right, name)


@typemap
def not_equal(left, right, name=''):
    '''
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Example:
        >>> C.not_equal([41., 42., 43.], [42., 42., 42.]).eval()
        array([ 1.,  0.,  1.], dtype=float32)

        >>> C.not_equal([-1,0,1], [0]).eval()
        array([ 1.,  0.,  1.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import not_equal
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return not_equal(left, right, name)


@typemap
def less_equal(left, right, name=''):
    '''
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Example:
        >>> C.less_equal([41., 42., 43.], [42., 42., 42.]).eval()
        array([ 1.,  1.,  0.], dtype=float32)

        >>> C.less_equal([-1,0,1], [0]).eval()
        array([ 1.,  1.,  0.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import less_equal
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return less_equal(left, right, name)

##########################################################################
# linear ops
##########################################################################

# This is a helper to wrap associative operations like plus(left, right) such
# that they accept multiple arguments.
def associative_multi_arg(f):
    '''
    The output of this operation is the result of an operation (`plus`, `log_add_exp`, `element_times`, `element_max`, `element_min`)
    of two or more input tensors. Broadcasting is supported.

    Example:
        >>> C.plus([1, 2, 3], [4, 5, 6]).eval()
        array([ 5.,  7.,  9.], dtype=float32)

        >>> C.element_times([5., 10., 15., 30.], [2.]).eval()
        array([ 10.,  20.,  30.,  60.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10], [3, 2, 3, 2, 3], [-13], [+42], 'multi_arg_example').eval()
        array([ 37.,  37.,  39.,  39.,  41.], dtype=float32)

        >>> C.element_times([5., 10., 15., 30.], [2.], [1., 2., 1., 2.]).eval()
        array([  10.,   40.,   30.,  120.], dtype=float32)

        >>> a = np.arange(3,dtype=np.float32)
        >>> np.exp(C.log_add_exp(np.log(1+a), np.log(1+a*a)).eval())
        array([ 2.,  4.,  8.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from functools import wraps
    @wraps(f)
    def associative_binary_operation(arg1, arg2, *more_args, **name_kwarg):
        name = (lambda name='': name)(**name_kwarg) # Python 2.7 does not allow (arg1, arg2, *more, name='')
        # in case name is specified without keyword
        if not name and more_args and isinstance(more_args[-1], str):
            name = more_args[-1]
            more_args = more_args[0:-1]
        # implement as a tree reduction
        def tree_reduce(args, name):
            n = len(args)
            if   n > 2:  return f(tree_reduce(args[:n//2], name=''),tree_reduce(args[n//2:], name=''), name=name) # only the outer-most op gets the 'name' parameter
            elif n == 2: return f(args[0],args[1], name=name)
            else:        return args[0]
        return tree_reduce((arg1, arg2) + more_args, name=name)
    return associative_binary_operation

@associative_multi_arg
@typemap
def plus(left, right, name=''):
    '''
    The output of this operation is the sum of the two or more input tensors. It supports broadcasting.

    Example:
        >>> C.plus([1, 2, 3], [4, 5, 6]).eval()
        array([ 5.,  7.,  9.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10]).eval()
        array([ 5.,  6.,  7.,  8.,  9.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10], [3, 2, 3, 2, 3], [-13], [+42], 'multi_arg_example').eval()
        array([ 37.,  37.,  39.,  39.,  41.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10], [3, 2, 3, 2, 3]).eval()
        array([  8.,   8.,  10.,  10.,  12.], dtype=float32)

    Args:
        arg1: left side tensor
        arg2: right side tensor
        *more_args: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import plus as cntk_py_plus
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_plus(left, right, name)


@typemap
def minus(left, right, name=''):
    '''
    The output of this operation is left minus right tensor. It supports broadcasting.

    Example:
        >>> C.minus([1, 2, 3], [4, 5, 6]).eval()
        array([-3., -3., -3.], dtype=float32)

        >>> C.minus([[1,2],[3,4]], 1).eval()
        array([[ 0.,  1.],
               [ 2.,  3.]], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from cntk.cntk_py import minus
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return minus(left, right, name)

@typemap
def pow(base, exponent, name=''):
    '''
    Computes `base` raised to the power of `exponent`. It supports broadcasting.
    This is well defined if `base` is non-negative or `exponent` is an integer.
    Otherwise the result is NaN. The gradient with respect to the base is  well
    defined if the forward operation is well defined. The gradient with respect
    to the exponent is well defined if the base is non-negative, and it is set
    to 0 otherwise.

    Example:
        >>> C.pow([1, 2, -2], [3, -2, 3]).eval()
        array([ 1.  ,  0.25, -8.  ], dtype=float32)

        >>> C.pow([[0.5, 2],[4, 1]], -2).eval()
        array([[ 4.    ,  0.25  ],
               [ 0.0625,  1.    ]], dtype=float32)

    Args:
        base: base tensor
        exponent: exponent tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from cntk.cntk_py import pow
    dtype = get_data_type(base, exponent)
    base = sanitize_input(base, dtype)
    exponent = sanitize_input(exponent, dtype)
    return pow(base, exponent, name)

@associative_multi_arg
@typemap
def element_times(left, right, name=''):
    '''
    The output of this operation is the element-wise product of the two or more input
    tensors. It supports broadcasting.

    Example:
        >>> C.element_times([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]).eval()
        array([ 0.5  ,  0.25 ,  0.125,  0.   ], dtype=float32)

        >>> C.element_times([5., 10., 15., 30.], [2.]).eval()
        array([ 10.,  20.,  30.,  60.], dtype=float32)

        >>> C.element_times([5., 10., 15., 30.], [2.], [1., 2., 1., 2.]).eval()
        array([  10.,   40.,   30.,  120.], dtype=float32)

    Args:
        arg1: left side tensor
        arg2: right side tensor
        *more_args: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_times as cntk_py_element_times
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_element_times(left, right, name)

@associative_multi_arg
@typemap
def element_max(left, right, name=''):
    '''
    The output of this operation is the element-wise max of the two or more input
    tensors. It supports broadcasting.

    Args:
        arg1: left side tensor
        arg2: right side tensor
        *more_args: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_max as cntk_py_element_max
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_element_max(left, right, name)

@associative_multi_arg
@typemap
def element_min(left, right, name=''):
    '''
    The output of this operation is the element-wise min of the two or more input
    tensors. It supports broadcasting.

    Args:
        arg1: left side tensor
        arg2: right side tensor
        *more_args: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_min as cntk_py_element_min
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_element_min(left, right, name)

@typemap
def element_divide(left, right, name=''):
    '''
    The output of this operation is the element-wise division of the two input
    tensors. It supports broadcasting.

    Example:
        >>> C.element_divide([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]).eval()
        array([ 2.,  4.,  8.,  0.], dtype=float32)

        >>> C.element_divide([5., 10., 15., 30.], [2.]).eval()
        array([  2.5,   5. ,   7.5,  15. ], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_divide
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return element_divide(left, right, name)


@associative_multi_arg
@typemap
def log_add_exp(left, right, name=''):
    '''
    Calculates the log of the sum of the exponentials
    of the two or more input tensors. It supports broadcasting.

    Example:
        >>> a = np.arange(3,dtype=np.float32)
        >>> np.exp(C.log_add_exp(np.log(1+a), np.log(1+a*a)).eval())
        array([ 2.,  4.,  8.], dtype=float32)
        >>> np.exp(C.log_add_exp(np.log(1+a), [0.]).eval())
        array([ 2.,  3.,  4.], dtype=float32)

    Args:
        arg1: left side tensor
        arg2: right side tensor
        *more_args: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import log_add_exp as cntk_py_log_add_exp
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_log_add_exp(left, right, name)

INFINITELY_REPEAT = cntk_py.MinibatchSource.infinitely_repeat

@typemap
def times(left, right, output_rank=1, infer_input_rank_to_map=TIMES_NO_INFERRED_INPUT_RANK, name=''):
    '''
    The output of this operation is the matrix product of the two input matrices.
    It supports broadcasting. Sparse is supported in the left operand, if it is a matrix.
    The operator '@' has been overloaded such that in Python 3.5 and later X @ W equals times(X, W).

    For better performance on times operation on sequence which is followed by sequence.reduce_sum, use
    infer_input_rank_to_map=TIMES_REDUCE_SEQUENCE_AXIS_WITHOUT_INFERRED_INPUT_RANK, i.e. replace following::

        sequence.reduce_sum(times(seq1, seq2))

    with::

        times(seq1, seq2, infer_input_rank_to_map=TIMES_REDUCE_SEQUENCE_AXIS_WITHOUT_INFERRED_INPUT_RANK)

    Example:
        >>> C.times([[1,2],[3,4]], [[5],[6]]).eval()
        array([[ 17.],
               [ 39.]], dtype=float32)

        >>> C.times(1.*np.reshape(np.arange(8), (2,2,2)),1.*np.reshape(np.arange(8), (2,2,2)), output_rank=1).eval()
        array([[ 28.,  34.],
               [ 76.,  98.]])

        >>> C.times(1.*np.reshape(np.arange(8), (2,2,2)),1.*np.reshape(np.arange(8), (2,2,2)), output_rank=2).eval()
        array([[[[  4.,   5.],
                 [  6.,   7.]],
        <BLANKLINE>
                [[ 12.,  17.],
                 [ 22.,  27.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 20.,  29.],
                 [ 38.,  47.]],
        <BLANKLINE>
                [[ 28.,  41.],
                 [ 54.,  67.]]]])

    Args:
        left: left side matrix or tensor
        right: right side matrix or tensor
        output_rank (int): in case we have tensors as arguments, output_rank represents
            the number of axes to be collapsed in order to transform the tensors
            into matrices, perform the operation and then reshape back (explode the axes)
        infer_input_rank_to_map (int): meant for internal use only. Always use default value
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import times
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return times(right, left, output_rank, infer_input_rank_to_map, name)


@typemap
def times_transpose(left, right, name=''):
    '''
    The output of this operation is the product of the first (``left``) argument with the second (``right``) argument transposed.
    The second (``right``) argument must have a rank of 1 or 2.
    This operation is conceptually computing ``np.dot(left, right.T)`` except when ``right`` is a vector
    in which case the output is ``np.dot(left,np.reshape(right,(1,-1)).T)`` (matching numpy when ``left`` is a vector).

    Example:
        >>> a=np.array([[1,2],[3,4]],dtype=np.float32)
        >>> b=np.array([2,-1],dtype=np.float32)
        >>> c=np.array([[2,-1]],dtype=np.float32)
        >>> d=np.reshape(np.arange(24,dtype=np.float32),(4,3,2))
        >>> print(C.times_transpose(a, a).eval())
        [[  5.  11.]
         [ 11.  25.]]
        >>> print(C.times_transpose(a, b).eval())
        [[ 0.]
         [ 2.]]
        >>> print(C.times_transpose(a, c).eval())
        [[ 0.]
         [ 2.]]
        >>> print(C.times_transpose(b, a).eval())
        [ 0.  2.]
        >>> print(C.times_transpose(b, b).eval())
        [ 5.]
        >>> print(C.times_transpose(b, c).eval())
        [ 5.]
        >>> print(C.times_transpose(c, a).eval())
        [[ 0.  2.]]
        >>> print(C.times_transpose(c, b).eval())
        [[ 5.]]
        >>> print(C.times_transpose(c, c).eval())
        [[ 5.]]
        >>> print(C.times_transpose(d, a).eval())
        [[[   2.    4.]
          [   8.   18.]
          [  14.   32.]]
        <BLANKLINE>
         [[  20.   46.]
          [  26.   60.]
          [  32.   74.]]
        <BLANKLINE>
         [[  38.   88.]
          [  44.  102.]
          [  50.  116.]]
        <BLANKLINE>
         [[  56.  130.]
          [  62.  144.]
          [  68.  158.]]]
        >>> print(C.times_transpose(d, b).eval())
        [[[ -1.]
          [  1.]
          [  3.]]
        <BLANKLINE>
         [[  5.]
          [  7.]
          [  9.]]
        <BLANKLINE>
         [[ 11.]
          [ 13.]
          [ 15.]]
        <BLANKLINE>
         [[ 17.]
          [ 19.]
          [ 21.]]]
        >>> print(C.times_transpose(d, c).eval())
        [[[ -1.]
          [  1.]
          [  3.]]
        <BLANKLINE>
         [[  5.]
          [  7.]
          [  9.]]
        <BLANKLINE>
         [[ 11.]
          [ 13.]
          [ 15.]]
        <BLANKLINE>
         [[ 17.]
          [ 19.]
          [ 21.]]]

    Args:
        left: left side tensor
        right: right side matrix or vector
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import times_transpose
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    rshape = sanitize_shape(right.shape)
    right = sanitize_input(right, dtype, (1,rshape[0]) if len(rshape) == 1 else None)
    return times_transpose(right, left, 1, name)

##########################################################################
# non_diff ops
##########################################################################

@typemap
def floor(arg, name=''):
    '''
    The output of this operation is the element wise value rounded to the largest
    integer less than or equal to the input.

    Example:
        >>> C.floor([0.2, 1.3, 4., 5.5, 0.0]).eval()
        array([ 0.,  1.,  4.,  5.,  0.], dtype=float32)

        >>> C.floor([[0.6, 3.3], [1.9, 5.6]]).eval()
        array([[ 0.,  3.],
               [ 1.,  5.]], dtype=float32)

        >>> C.floor([-5.5, -4.2, -3., -0.7, 0]).eval()
        array([-6., -5., -3., -1.,  0.], dtype=float32)

        >>> C.floor([[-0.6, -4.3], [1.9, -3.2]]).eval()
        array([[-1., -5.],
               [ 1., -4.]], dtype=float32)

    Args:
        arg: input tensor
        name (str, optional): the name of the Function instance in the network (optional)
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import floor
    arg = sanitize_input(arg, get_data_type(arg))
    return floor(arg, name)


@typemap
def ceil(arg, name=''):
    '''
    The output of this operation is the element wise value rounded to the smallest
    integer greater than or equal to the input.

    Example:
        >>> C.ceil([0.2, 1.3, 4., 5.5, 0.0]).eval()
        array([ 1.,  2.,  4.,  6.,  0.], dtype=float32)

        >>> C.ceil([[0.6, 3.3], [1.9, 5.6]]).eval()
        array([[ 1.,  4.],
               [ 2.,  6.]], dtype=float32)

    Args:
        arg: input tensor
        name (str, optional): the name of the Function instance in the network (optional)
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import ceil
    arg = sanitize_input(arg, get_data_type(arg))
    return ceil(arg, name)


@typemap
def round(arg, name=''):
    '''
    The output of this operation is the element wise value rounded to the nearest integer.
    In case of tie, where element can have exact fractional part of 0.5
    this operation follows "round half-up" tie breaking strategy.
    This is different from the round operation of numpy which follows
    round half to even.

    Example:
        >>> C.round([0.2, 1.3, 4., 5.5, 0.0]).eval()
        array([ 0.,  1.,  4.,  6.,  0.], dtype=float32)

        >>> C.round([[0.6, 3.3], [1.9, 5.6]]).eval()
        array([[ 1.,  3.],
               [ 2.,  6.]], dtype=float32)

        >>> C.round([-5.5, -4.2, -3., -0.7, 0]).eval()
        array([-5., -4., -3., -1.,  0.], dtype=float32)

        >>> C.round([[-0.6, -4.3], [1.9, -3.2]]).eval()
        array([[-1., -4.],
               [ 2., -3.]], dtype=float32)

    Args:
        arg: input tensor
        name (str, optional): the name of the Function instance in the network (optional)
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import round
    arg = sanitize_input(arg, get_data_type(arg))
    return round(arg, name)


##########################################################################
# non_linear and nn ops
##########################################################################

@typemap
def clip(x, min_value, max_value, name=''):
    '''
    Computes a tensor with all of its values clipped to fall
    between ``min_value`` and ``max_value``, i.e.
    ``min(max(x, min_value), max_value)``.

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.clip([1., 2.1, 3.0, 4.1], 2., 4.).eval()
        array([ 2. ,  2.1,  3. ,  4. ], dtype=float32)

        >>> C.clip([-10., -5., 0., 5., 10.], [-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.]).eval()
        array([-5., -4.,  0.,  4.,  9.], dtype=float32)

    Args:
        x: tensor to be clipped
        min_value (float): a scalar or a tensor which represents the minimum value to clip element
         values to
        max_value (float): a scalar or a tensor which represents the maximum value to clip element
         values to
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import clip
    x = sanitize_input(x, get_data_type(x))
    min_value = sanitize_input(min_value, get_data_type(min_value))
    max_value = sanitize_input(max_value, get_data_type(max_value))
    return clip(x, min_value, max_value, name)


@typemap
def relu(x, name=''):
    '''
    Rectified linear operation. Computes the element-wise rectified linear
    of ``x``: ``max(x, 0)``

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.relu([[-1, -0.5, 0, 1, 2]]).eval()
        array([[ 0.,  0.,  0.,  1.,  2.]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import re_lu
    x = sanitize_input(x)
    return re_lu(x, name)


@typemap
def elu(x, alpha=1.0, name=''):
    '''
    Exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``max(x, 0)`` for ``x >= 0`` and ``x``: ``alpha * (exp(x)-1)`` otherwise.

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.elu([[-1, -0.5, 0, 1, 2]]).eval()
        array([[-0.632121, -0.393469,  0.      ,  1.      ,  2.      ]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        name (`str`, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import elu
    x = sanitize_input(x)
    return elu(x, alpha, name)

@typemap
def selu(x, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=''):
    '''
    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.selu([[-1, -0.5, 0, 1, 2]]).eval()
        array([[-1.111331, -0.691758,  0.      ,  1.050701,  2.101402]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        name (`str`, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import selu
    x = sanitize_input(x)
    return selu(x, scale, alpha, name)


@typemap
def leaky_relu(x, alpha = 0.01, name=''):
    '''
    Leaky Rectified linear operation. Computes the element-wise leaky rectified linear
    of ``x``: ``max(x, 0)`` for ``x >= 0`` and ``x``: ``alpha*x`` otherwise.

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.leaky_relu([[-1, -0.5, 0, 1, 2]]).eval()
        array([[-0.01 , -0.005,  0.   ,  1.   ,  2.   ]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        alpha (float): the alpha term of the above equation.
        name (`str`, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import leaky_re_lu
    x = sanitize_input(x)
    return leaky_re_lu(x, alpha, name)

@typemap
def param_relu(alpha, x, name=''):
    '''
    Parametric rectified linear operation. Computes the element-wise parameteric rectified linear
    of ``x``: ``max(x, 0)`` for ``x >= 0`` and ``x``: ``alpha*x`` otherwise.

    The output tensor has the same shape as ``x``.

    Example:
        >>> alpha = C.constant(value=[[0.5, 0.5, 0.5, 0.5, 0.5]])
        >>> C.param_relu(alpha, [[-1, -0.5, 0, 1, 2]]).eval()
        array([[-0.5 , -0.25,  0.  ,  1.  ,  2.  ]], dtype=float32)

    Args:
        alpha (:class:`~cntk.variables.Parameter`): same shape as x
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        name (`str`, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import pre_lu
    x = sanitize_input(x)
    return pre_lu(alpha, x, name)

@typemap
def straight_through_impl(x, name=''):
    '''
    element-wise binarization node using the straight through estimator

    Example:
        >>> # create (1,3) matrix 
        >>> data = np.asarray([[-3, 4, -2]], dtype=np.float32)
        >>> x = C.input_variable((3))
        >>> C.straight_through_impl(x).eval({x:data})
        array([[-1., 1., -1.]], dtype=float32)

    Args:
        inputs: one input tensor
        name (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from cntk.cntk_py import straight_through
    x = sanitize_input(x)
    return straight_through(x, name) # C++ projection expects inputs as a list

@typemap
def softplus(x, steepness=1, name=''):
    '''
    Softplus operation. Computes the element-wise softplus of ``x``:

    :math:`\\mathrm{softplus}(x) = {\\log(1+\\exp(x))}`

    The optional ``steepness`` allows to make the knee sharper (``steepness>1``) or softer, by computing
    ``softplus(x * steepness) / steepness``.
    (For very large steepness, this approaches a linear rectifier).

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.softplus([[-1, -0.5, 0, 1, 2]]).eval()
        array([[ 0.313262,  0.474077,  0.693147,  1.313262,  2.126928]], dtype=float32)

        >>> C.softplus([[-1, -0.5, 0, 1, 2]], steepness=4).eval()
        array([[ 0.004537,  0.031732,  0.173287,  1.004537,  2.000084]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        steepness (float, optional): optional steepness factor
        name (`str`, default to ''): the name of the Function instance in the network
    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import softplus
    x = sanitize_input(x)
    if steepness == 1:
        return softplus(x, name)
    xp = placeholder()
    f = typemap(softplus)(steepness * xp) / steepness
    return as_block(f, [(xp, x)], 'softplus', name)

@typemap
def softsign(x, steepness=1, name=''):
    '''
    Computes the element-wise softsign of ``x``:

    :math:`sigmoid(x) = {x \\over {1+\\abs(x)}}`

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.softsign([[-1, 0, 1]]).eval()
        array([[-0.5,  0. ,  0.5]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import softsign
    x = sanitize_input(x)
    return softsign(x, name)

@typemap
def sigmoid(x, name=''):
    '''
    Computes the element-wise sigmoid of ``x``:

    :math:`sigmoid(x) = {1 \\over {1+\\exp(-x)}}`

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.sigmoid([-2, -1., 0., 1., 2.]).eval()
        array([ 0.119203,  0.268941,  0.5     ,  0.731059,  0.880797], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sigmoid
    x = sanitize_input(x)
    return sigmoid(x, name)


@typemap
def tanh(x, name=''):
    '''
    Computes the element-wise tanh of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> C.tanh([[1,2],[3,4]]).eval()
        array([[ 0.761594,  0.964028],
               [ 0.995055,  0.999329]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import tanh
    x = sanitize_input(x)
    return tanh(x, name)

@typemap
def atanh(x, name=''):
    '''
    Computes the element-wise atanh of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.atanh([[0.9,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 1.47222,  0.54931],
               [-0.25541, -0.97296]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import atanh
    x = sanitize_input(x)
    return atanh(x, name)

@typemap
def sin(x, name=''):
    '''
    Computes the element-wise sine of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.sin(np.arcsin([[1,0.5],[-0.25,-0.75]])).eval(),5)
        array([[ 1.  ,  0.5 ],
               [-0.25, -0.75]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sin
    x = sanitize_input(x)
    return sin(x, name)

@typemap
def cos(x, name=''):
    '''
    Computes the element-wise cosine of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.cos(np.arccos([[1,0.5],[-0.25,-0.75]])).eval(),5)
        array([[ 1.  ,  0.5 ],
               [-0.25, -0.75]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import cos
    x = sanitize_input(x)
    return cos(x, name)

@typemap
def tan(x, name=''):
    '''
    Computes the element-wise tangent of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.tan([-1, 0, 1]).eval(), 5)
        array([-1.55741,  0.     ,  1.55741], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import tan
    x = sanitize_input(x)
    return tan(x, name)

@typemap
def acos(x, name=''):
    '''
    Computes the element-wise arccos (inverse cosine) of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.acos([[1,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 0.     ,  1.0472 ],
               [ 1.82348,  2.41886]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import acos
    x = sanitize_input(x)
    return acos(x, name)

@typemap
def asin(x, name=''):
    '''
    Computes the element-wise arcsin (inverse sine) of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.asin([[1,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 1.5708 ,  0.5236 ],
               [-0.25268, -0.84806]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import asin
    x = sanitize_input(x)
    return asin(x, name)

@typemap
def atan(x, name=''):
    '''
    Computes the element-wise arctan (inverse tangent) of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.atan([-1, 0, 1]).eval(), 5)
        array([-0.78539997,  0.        ,  0.78539997], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import atan
    x = sanitize_input(x)
    return atan(x, name)

@typemap
def sinh(x, name=''):
    '''
    Computes the element-wise sinh of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.sinh([[1,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 1.1752 ,  0.5211 ],
               [-0.25261, -0.82232]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sinh
    x = sanitize_input(x)
    return sinh(x, name)

@typemap
def cosh(x, name=''):
    '''
    Computes the element-wise cosh of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.cosh([[1,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 1.54308,  1.12763],
               [ 1.03141,  1.29468]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import cosh
    x = sanitize_input(x)
    return cosh(x, name)

@typemap
def asinh(x, name=''):
    '''
    Computes the element-wise asinh of ``x``:

    The output tensor has the same shape as ``x``.

    Example:
        >>> np.round(C.asinh([[1,0.5],[-0.25,-0.75]]).eval(),5)
        array([[ 0.88137,  0.48121],
               [-0.24747, -0.69315]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import asinh
    x = sanitize_input(x)
    return asinh(x, name)

@typemap
def log_softmax(x, axis = None, name = ''):
    '''
    Computes the logsoftmax normalized values of x. That is, y = x - log(reduce_sum(exp(x), axis))
    (the implementation uses an equivalent formula for numerical stability).

    It is also possible to use `x - reduce_log_sum_exp(x, axis)` instead of log_softmax:
    this can be faster (one reduce pass instead of two), but can behave slightly differently numerically.

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        axis (int): axis along which the logsoftmax operation will be performed (the default is the last axis)
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import log_softmax
    x = sanitize_input(x)
    if axis is not None:
        axis = sanitize_axis(axis)
        return log_softmax(x, axis, name)
    else:
        return log_softmax(x, name)

@typemap
def softmax(x, axis=None, name=''):
    r'''
    Computes the gradient of :math:`f(z)=\log\sum_i\exp(z_i)` at ``z = x``. Concretely,

    :math:`\mathrm{softmax}(x)=\left[\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\ldots\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\right]`

    with the understanding that the implementation can use equivalent formulas
    for efficiency and numerical stability.

    The output is a vector of non-negative numbers that sum to 1 and can
    therefore be interpreted as probabilities for mutually exclusive outcomes
    as in the case of multiclass classification.

    If ``axis`` is given as integer, then the softmax will be computed along that axis.
    If the provided ``axis`` is -1, it will be computed along the last axis. Otherwise,
    softmax will be applied to all axes.

    Example:
        >>> C.softmax([[1, 1, 2, 3]]).eval()
        array([[ 0.082595,  0.082595,  0.224515,  0.610296]], dtype=float32)

        >>> C.softmax([1, 1]).eval()
        array([ 0.5,  0.5], dtype=float32)

        >>> C.softmax([[[1, 1], [3, 5]]], axis=-1).eval()
        array([[[ 0.5     ,  0.5     ],
                [ 0.119203,  0.880797]]], dtype=float32)

        >>> C.softmax([[[1, 1], [3, 5]]], axis=1).eval()
        array([[[ 0.119203,  0.017986],
                [ 0.880797,  0.982014]]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the softmax operation will be performed
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import softmax
    x = sanitize_input(x)
    if axis is not None:
        axis = sanitize_axis(axis)
        return softmax(x, axis, name)
    else:
        return softmax(x, name)

@typemap
def hardmax(x, name=''):
    '''
    Creates a tensor with the same shape as the input tensor, with zeros everywhere and a 1.0 where the
    maximum value of the input tensor is located. If the maximum value is repeated, 1.0 is placed in the first location found.

    Example:
        >>> C.hardmax([1., 1., 2., 3.]).eval()
        array([ 0.,  0.,  0.,  1.], dtype=float32)

        >>> C.hardmax([1., 3., 2., 3.]).eval()
        array([ 0.,  1.,  0.,  0.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import hardmax
    x = sanitize_input(x)
    return hardmax(x, name)


@typemap
def top_k(x, k, axis=-1, name=''):
    '''
    Computes the ``k`` largest values of the input tensor and the corresponding indices
    along the specified axis (default the last axis). The returned 
    :class:`~cntk.ops.functions.Function` has two outputs. The first one 
    contains the top ``k`` values in sorted order, and the second one contains 
    the corresponding top ``k`` indices.

    Example:
        >>> x = C.input_variable(10)
        >>> y = C.top_k(-x * C.log(x), 3)
        >>> x0 = np.arange(10,dtype=np.float32)*0.1
        >>> top = y.eval({x:x0})
        >>> top_values = top[y.outputs[0]]
        >>> top_indices = top[y.outputs[1]]
        >>> top_indices
        array([[ 4.,  3.,  5.]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        k (int): number of top items to return
        axis: axis along which to perform the operation (default: -1)
        name (str): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import top_k
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return top_k(x, k, axis, name)

@typemap
def hard_sigmoid(x, alpha, beta, name = ''):
    '''
    Computes the element-wise HardSigmoid function, y = max(0, min(1, alpha * x + beta)).

    Example:
        >>> alpha = 1 
        >>> beta = 2
        >>> C.hard_sigmoid([-2.5, -1.5, 1], alpha, beta).eval()
        array([ 0. ,  0.5,  1. ], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        alpha (float): the alpha term of the above equation.
        beta (float): the beta term of the above equation.
        name (str): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import hard_sigmoid
    x = sanitize_input(x)
    return hard_sigmoid(x, alpha, beta, name)

@typemap
def exp(x, name=''):
    '''
    Computes the element-wise exponential of ``x``:

    :math:`\\exp(x) = {e^x}`

    Example:
        >>> C.exp([0., 1.]).eval()
        array([ 1.      ,  2.718282], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import exp
    x = sanitize_input(x)
    return exp(x, name)


@typemap
def log(x, name=''):
    '''
    Computes the element-wise the natural logarithm of ``x``:

    Example:
        >>> C.log([1., 2.]).eval()
        array([ 0.      ,  0.693147], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`

    Note:
        CNTK returns -85.1 for log(x) if ``x`` is negative or zero. The reason is that
        it uses 1e-37 (whose natural logarithm is -85.1) as the smallest float
        number for `log`, because this is the only guaranteed precision across
        platforms. This will be changed to return `NaN` and `-inf`.
    '''
    from cntk.cntk_py import log
    x = sanitize_input(x)
    return log(x, name)


@typemap
def sqrt(x, name=''):
    '''
    Computes the element-wise square-root of ``x``:

    :math:`sqrt(x) = {\\sqrt[2]{x}}`

    Example:
        >>> C.sqrt([0., 4.]).eval()
        array([ 0.,  2.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`

    Note:
        CNTK returns zero for sqrt of negative nubmers, this will be changed to
        return NaN
    '''
    from cntk.cntk_py import sqrt
    x = sanitize_input(x)
    return sqrt(x, name)


@typemap
def square(x, name=''):
    '''
    Computes the element-wise square of ``x``:

    Example:
        >>> C.square([1., 10.]).eval()
        array([   1.,  100.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import square
    x = sanitize_input(x)
    return square(x, name)


@typemap
def abs(x, name=''):
    '''
    Computes the element-wise absolute of ``x``:

    :math:`abs(x) = |x|`

    Example:
        >>> C.abs([-1, 1, -2, 3]).eval()
        array([ 1.,  1.,  2.,  3.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import abs
    x = sanitize_input(x)
    return abs(x, name)

@typemap
def element_and(x, y, name=''):
    '''
    Computes the element-wise logic AND of ``x``.

    Example:
        >>> C.element_and([1, 1, 0, 0], [1, 0, 1, 0]).eval()
        array([ 1.,  0.,  0.,  0.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_and
    x = sanitize_input(x)
    y = sanitize_input(y)
    return element_and(x, y, name)

@typemap
def element_not(x, name=''):
    '''
    Computes the element-wise logic NOT of ``x`` and ``y``.

    Example:
        >>> C.element_not([1, 1, 0, 0]).eval()
        array([ 0.,  0.,  1.,  1.], dtype=float32)

    Args:
        x, y: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_not
    x = sanitize_input(x)
    return element_not(x, name)

@typemap
def element_or(x, y, name=''):
    '''
    Computes the element-wise logic OR of ``x`` and ``y``.

    Example:
        >>> C.element_or([1, 1, 0, 0], [1, 0, 1, 0]).eval()
        array([ 1.,  1.,  1.,  0.], dtype=float32)

    Args:
        x, y: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_or
    x = sanitize_input(x)
    y = sanitize_input(y)
    return element_or(x, y, name)

@typemap
def element_xor(x, y, name=''):
    '''
    Computes the element-wise logic XOR of ``x`` and ``y``.

    Example:
        >>> C.element_xor([1, 1, 0, 0], [1, 0, 1, 0]).eval()
        array([ 0.,  1.,  1.,  0.], dtype=float32)

    Args:
        x, y: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_xor
    x = sanitize_input(x)
    y = sanitize_input(y)
    return element_xor(x, y, name)

@typemap
def negate(x, name=''):
    '''
    Computes the element-wise negation of ``x``:

    :math:`negate(x) = -x`

    Example:
        >>> C.negate([-1, 1, -2, 3]).eval()
        array([ 1., -1.,  2., -3.], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import negate
    x = sanitize_input(x)
    return negate(x, name)


@typemap
def reciprocal(x, name=''):
    '''
    Computes the element-wise reciprocal of ``x``:

    Example:
        >>> C.reciprocal([-1/3, 1/5, -2, 3]).eval()
        array([-3.      ,  5.      , -0.5     ,  0.333333], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reciprocal
    x = sanitize_input(x)
    return reciprocal(x, name)


@typemap
def ones_like(x, name=''):
    '''
    Creates an all-ones tensor with the same shape and dynamic axes as ``x``:

    Example:
        >>> x0 = np.arange(24).reshape((2, 3, 4)).astype('f')
        >>> x = C.input_variable((3, 4))
        >>> C.ones_like(x).eval({x: x0})
        array([[[ 1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.]]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import ones_like
    x = sanitize_input(x)
    return ones_like(x, name)


@typemap
def zeros_like(x, name=''):
    '''
    Creates an all-zeros tensor with the same shape and dynamic axes as ``x``:

    Example:
        >>> x0 = np.arange(24).reshape((2, 3, 4)).astype('f')
        >>> x = C.input_variable((3, 4))
        >>> C.zeros_like(x).eval({x: x0})
        array([[[ 0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.]],
        <BLANKLINE>
               [[ 0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.]]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import zeros_like
    x = sanitize_input(x)
    return zeros_like(x, name)


@typemap
def eye_like(x, sparse_output = True, name=''):
    '''
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a matrix,
     ``x`` must have exactly two axes (counting both dynamic and static axes).

    Example:
        >>> x0 = np.arange(12).reshape((3, 4)).astype('f')
        >>> x = C.input_variable(4)
        >>> C.eye_like(x).eval({x: x0}).toarray()
        array([[ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor of rank 2
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import eye_like
    x = sanitize_input(x)
    if len(x.dynamic_axes) + len(x.shape) != 2:
        raise(ValueError('eye_like operand must have exactly two axes (counting both dynamic and static axes) however "%s" is provided as the operand'%x))
    if any([ax.is_sequence_axis for ax in x.dynamic_axes]):
        raise (ValueError(
            'eye_like operand must have no sequence axis however "%s" is provided as the operand' % x))
    return eye_like(x, sparse_output, name)


@typemap
def element_select(flag, value_if_true, value_if_false, name=''):
    '''
    return either ``value_if_true`` or ``value_if_false`` based on the value of ``flag``.
    If ``flag`` != 0 ``value_if_true`` is returned, otherwise ``value_if_false``.
    Behaves analogously to numpy.where(...).

    Example:
        >>> C.element_select([-10, -1, 0, 0.3, 100], [1, 10, 100, 1000, 10000], [ 2, 20, 200, 2000, 20000]).eval()
        array([     1.,     10.,    200.,   1000.,  10000.], dtype=float32)

    Args:
        flag: condition tensor
        value_if_true: true branch tensor
        value_if_false: false branch tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import element_select
    flag = sanitize_input(flag)
    value_if_true = sanitize_input(value_if_true)
    value_if_false = sanitize_input(value_if_false)
    return element_select(flag, value_if_true, value_if_false, name)


# TODO: does this belong into .sequence?
@typemap
def optimized_rnnstack(operand, weights, hidden_size, num_layers,
                       bidirectional=False, recurrent_op='lstm', name=''):
    '''
    An RNN implementation that uses the primitives in cuDNN.
    If cuDNN is not available it fails. You can use :class:`~cntk.misc.optimized_rnnstack_converter.convert_optimized_rnnstack`
    to convert a model to GEMM-based implementation when no cuDNN.

    Args:
        operand: input of the optimized RNN stack.
        weights: parameter tensor that holds the learned weights.
        hidden_size (int): number of hidden units in each layer (and in each direction).
        num_layers (int): number of layers in the stack.
        bidirectional(bool, default False): whether each layer should compute both in forward
         and separately in backward mode and concatenate the results
         (if True the output is twice the hidden_size). The default is
         False which means the recurrence is only computed in the forward direction.
        recurrent_op (str, optional): one of 'lstm', 'gru', 'relu', or 'tanh'.
        name (str, optional): the name of the Function instance in the network

    Example:
        >>> from _cntk_py import constant_initializer
        >>> W = C.parameter((C.InferredDimension,4), constant_initializer(0.1))
        >>> x = C.input_variable(shape=(4,))
        >>> s = np.reshape(np.arange(20.0, dtype=np.float32), (5,4))
        >>> t = np.reshape(np.arange(12.0, dtype=np.float32), (3,4))
        >>> f = C.optimized_rnnstack(x, W, 8, 2) # doctest: +SKIP
        >>> r = f.eval({x:[s,t]})                # doctest: +SKIP
        >>> len(r)                               # doctest: +SKIP
        2
        >>> print(*r[0].shape)                   # doctest: +SKIP
        5 8
        >>> print(*r[1].shape)                   # doctest: +SKIP
        3 8
        >>> r[0][:3,:]-r[1]                      # doctest: +SKIP
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    # FIXME figure out how to only SKIP the doctest in CPU
    from cntk.cntk_py import optimized_rnnstack
    operand = sanitize_input(operand)
    if recurrent_op not in set(['lstm','gru','rnnReLU','rnnTanh']):
        raise(ValueError('unsupported recurrent_op value "%s"'%recurrent_op))

    return optimized_rnnstack(operand, weights, hidden_size, num_layers,
                       bidirectional, recurrent_op, name)


##########################################################################
# reshaping ops
##########################################################################

# TODO: enable when it is exposed in c++


@typemap
def reshape(x, shape, begin_axis=None, end_axis=None, name=''):
    '''
    Reinterpret input samples as having different tensor dimensions
    One dimension may be specified as 0 and will be inferred

    The output tensor has the shape specified by 'shape'.

    Example:
        >>> i1 = C.input_variable(shape=(3,2))
        >>> C.reshape(i1, (2,3)).eval({i1:np.asarray([[[[0., 1.],[2., 3.],[4., 5.]]]], dtype=np.float32)})
        array([[[ 0.,  1.,  2.],
                 [ 3.,  4.,  5.]]], dtype=float32)

    Args:
        x: tensor to be reshaped
        shape (tuple): a tuple defining the resulting shape. The specified shape tuple
         may contain -1 for at most one axis, which is automatically inferred to the
         correct dimension size by dividing the total size of the sub-shape being reshaped
         with the product of the dimensions of all the non-inferred axes of the replacement
         shape.
        begin_axis (int or None): shape replacement begins at this axis. Negative values
         are counting from the end. `None` is the same as 0. To refer to the end of the shape tuple,
         pass `Axis.new_leading_axis()`.
        end_axis (int or None): shape replacement ends at this axis (excluding this axis).
         Negative values are counting from the end. `None` refers to the end of the shape tuple.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reshape
    x = sanitize_input(x)
    shape = sanitize_shape(shape)

    if begin_axis is None:
        begin_axis = Axis(0)

    if end_axis is None:
        end_axis = Axis.new_leading_axis()

    # Pass begin_axis as the end_axis and vice versa to account for
    # the automatic shape reversal across the python SWIG boundary
    def sanitize_reshape_axis(axis):
        if isinstance(axis, numbers.Integral):
            axis = Axis(axis)

        if not axis.is_static_axis:
            return axis

        if (axis ==  Axis.new_leading_axis()):
            return Axis(0)
        elif (axis == Axis(0)):
            return Axis.new_leading_axis()
        else:
            return Axis(-axis.static_axis_index())

    internal_reshape_begin_axis = sanitize_reshape_axis(end_axis)
    internal_reshape_end_axis = sanitize_reshape_axis(begin_axis)

    return reshape(x, shape, internal_reshape_begin_axis, internal_reshape_end_axis, name)

@typemap
def squeeze(x, axes=None, name=''):
    '''
        Removes axes whose size is 1. If ``axes`` is specified, and any of 
        their size is not 1 an exception will be raised.

    Example:
        >>> x0 = np.arange(12).reshape((2, 2, 1, 3)).astype('f')
        >>> x = C.input_variable((2, 1, 3))
        >>> C.squeeze(x).eval({x: x0})
        array([[[  0.,   1.,   2.],
                [  3.,   4.,   5.]],
        <BLANKLINE>
               [[  6.,   7.,   8.],
                [  9.,  10.,  11.]]], dtype=float32)

    Args:
        x: input tensor
        axes: The axes to squeeze out (default: all static axes).
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import squeeze
    x = sanitize_input(x)
    if axes is None:
        return squeeze(x, name)
    else:
        axes = sanitize_axis_list(axes)
        return squeeze(x, axes, name)


@typemap
def expand_dims(x, axis, name=''):
    '''
        Adds a singleton (size 1) axis at position ``axis``.

    Example:
        >>> x0 = np.arange(12).reshape((2, 2, 3)).astype('f')
        >>> x = C.input_variable((2, 3))
        >>> C.expand_dims(x, 0).eval({x: x0})
        array([[[[  0.,   1.,   2.]],
        <BLANKLINE>
                [[  3.,   4.,   5.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[  6.,   7.,   8.]],
        <BLANKLINE>
                [[  9.,  10.,  11.]]]], dtype=float32)

    Args:
        x: input tensor
        axis: The position to insert the singleton axis.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import expand_dims
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return expand_dims(x, axis, name)

@typemap
def transpose(x, perm, name=''):
    '''
    Permutes the axes of the tensor. The output has the same data but the axes
    are permuted according to ``perm``.

    Example:
        >>> a = np.arange(24).reshape(2,3,4).astype('f')
        >>> np.array_equal(C.transpose(a, perm=(2, 0, 1)).eval(), np.transpose(a, (2, 0, 1)))
        True

    Args:
        x: tensor to be transposed
        perm  (list): the permutation to apply to the axes.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import transpose
    x = sanitize_input(x)
    if type(perm) in [int, Axis]:
        raise TypeError('transpose expects a permutation; to swap two axes use swapaxes')
    perm = [Axis(p) for p in sanitize_permutation(perm)]
    return transpose(x, perm, name)


@typemap
def swapaxes(x, axis1=0, axis2=1, name=''):
    '''
    Swaps two axes of the tensor. The output tensor has the same data but with
    ``axis1`` and ``axis2`` swapped.

    Example:
        >>> C.swapaxes([[[0,1],[2,3],[4,5]]], 1, 2).eval()
        array([[[ 0.,  2.,  4.],
                [ 1.,  3.,  5.]]], dtype=float32)

    Args:
        x: tensor to be transposed
        axis1 (int or :class:`~cntk.axis.Axis`): the axis to swap with ``axis2``
        axis2 (int or :class:`~cntk.axis.Axis`): the axis to swap with ``axis1``
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import transpose_axes
    x = sanitize_input(x)
    axis1 = sanitize_axis(axis1)
    axis2 = sanitize_axis(axis2)
    return transpose_axes(x, axis1, axis2, name)


@typemap
def pad(x, pattern, mode=CONSTANT_PAD, constant_value = 0, name=''):
    '''
    Pads a tensor according to the specified patterns.
    Three padding modes are supported: CONSTANT / REFLECT / SYMMETRIC.

    Example:
        >>> data = np.arange(6, dtype=np.float32).reshape((2,3))
        >>> x = C.constant(value=data)
        >>> C.pad(x, pattern=[(1,1),(2,2)], mode=C.ops.CONSTANT_PAD, constant_value=1).eval()
        array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  0.,  1.,  2.,  1.,  1.],
               [ 1.,  1.,  3.,  4.,  5.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]], dtype=float32)
        >>> C.pad(x, pattern=[(1,1),(2,2)], mode=C.ops.REFLECT_PAD).eval()
        array([[ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
               [ 2.,  1.,  0.,  1.,  2.,  1.,  0.],
               [ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
               [ 2.,  1.,  0.,  1.,  2.,  1.,  0.]], dtype=float32)
        >>> C.pad(x, pattern=[(1,1),(2,2)], mode=C.ops.SYMMETRIC_PAD).eval()
        array([[ 1.,  0.,  0.,  1.,  2.,  2.,  1.],
               [ 1.,  0.,  0.,  1.,  2.,  2.,  1.],
               [ 4.,  3.,  3.,  4.,  5.,  5.,  4.],
               [ 4.,  3.,  3.,  4.,  5.,  5.,  4.]], dtype=float32)

    Args:
        x: tensor to be padded.
        pattern (list of tuple with 2 integers): how many values to add before and after the contents of the tensor in each dimension.
        mode (int): padding mode: C.ops.CONSTANT_PAD, C.ops.REFLECT_PAD and C.ops.SYMMETRIC_PAD
        constant_value: the value used to fill the padding cells, only meaningful under CONSTANT mode.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import pad
    if (any([len(_) != 2 for _  in pattern])):
        raise ValueError("padding pattern should be a integer list with shape [n, 2]")
    x = sanitize_input(x)
    head = [p[0] for p in reversed(pattern)]
    foot = [p[1] for p in reversed(pattern)]
    return pad(x, mode, head, foot, constant_value, name)


@typemap
def slice(x, axis, begin_index, end_index, strides=None, name=''):
    '''
    Slice the input along one or multiple axes.

    Example:
        >>> # slice using input variable
        >>> # create 2x3 matrix
        >>> x1 = C.input_variable((2,3))
        >>> # slice index 1 (second) at first axis
        >>> C.slice(x1, 0, 1, 2).eval({x1: np.asarray([[[1,2,-3],
        ...                                             [4, 5, 6]]],dtype=np.float32)})
        array([[[ 4.,  5.,  6.]]], dtype=float32)
        <BLANKLINE>
        >>> # slice index 0 (first) at second axis
        >>> C.slice(x1, 1, 0, 1).eval({x1: np.asarray([[[1,2,-3],
        ...                                             [4, 5, 6]]],dtype=np.float32)})
        array([[[ 1.],
                [ 4.]]], dtype=float32)
        >>> # slice with strides
        >>> C.slice(x1, 0, 0, 2, 2).eval({x1: np.asarray([[[1,2,-3],
        ...                                                [4, 5, 6]]],dtype=np.float32)})
        array([[[ 1.,  2., -3.]]], dtype=float32)
        <BLANKLINE>
        >>> # reverse
        >>> C.slice(x1, 0, 0, 2, -1).eval({x1: np.asarray([[[1,2,-3],
        ...                                                 [4, 5, 6]]],dtype=np.float32)})
        array([[[ 4.,  5.,  6.],
        [ 1.,  2., -3.]]], dtype=float32)
        <BLANKLINE>
        >>> # slice along multiple axes
        >>> C.slice(x1, [0,1], [1,0], [2,1]).eval({x1: np.asarray([[[1, 2, -3],
        ...                                                         [4, 5, 6]]],dtype=np.float32)})
        array([[[ 4.]]], dtype=float32)
        <BLANKLINE>
        >>> # slice using constant
        >>> data = np.asarray([[1, 2, -3],
        ...                    [4, 5,  6]], dtype=np.float32)
        >>> x = C.constant(value=data)
        >>> C.slice(x, 0, 1, 2).eval()
        array([[ 4.,  5.,  6.]], dtype=float32)
        >>> C.slice(x, 1, 0, 1).eval()
        array([[ 1.],
               [ 4.]], dtype=float32)
        >>> C.slice(x, [0,1], [1,0], [2,1]).eval()
        array([[ 4.]], dtype=float32)
        <BLANKLINE>
        >>> # slice using the index overload
        >>> data = np.asarray([[1, 2, -3],
        ...                    [4, 5,  6]], dtype=np.float32)
        >>> x = C.constant(value=data)
        >>> x[0].eval()
        array([[ 1.,  2.,  -3.]], dtype=float32)
        >>> x[0, [1,2]].eval()
        array([[ 2.,  -3.]], dtype=float32)
        <BLANKLINE>
        >>> x[1].eval()
        array([[ 4.,  5.,  6.]], dtype=float32)
        >>> x[:,:2,:].eval()
        array([[ 1.,  2.],
               [ 4.,  5.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which ``begin_index`` and ``end_index``
         will be used. If it is of type int it will be used as a static axis.
        begin_index (int): the index along axis where the slicing starts
        end_index (int): the index along axis where the slicing ends (exclusive)
        strides(int): step sizes when applying slice, negative value means in reverse order
        name (str, optional): the name of the Function instance in the network

    See also:
        Indexing in NumPy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import slice
    x = sanitize_input(x)
    axis = sanitize_axis_list(axis)
    begin_index = sanitize_shape(begin_index)
    end_index = sanitize_shape(end_index)
    if strides is None:
        strides = [1 for _ in axis]
    elif not type(strides) in (list, tuple):
        strides = [strides]
    return slice(x, axis, begin_index, end_index, strides, name)

# TODO: enable when it is exposed in c++


@typemap
def splice(*inputs, **kw_axis_name):
    '''
    Concatenate the input tensors along an axis.

    Example:
        >>> # create 2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data1 = np.asarray([[[1, 2],
        ...                      [4, 5]]], dtype=np.float32)

        >>> x = C.constant(value=data1)
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data2 = np.asarray([[[10, 20],
        ...                       [30, 40],
        ...                       [50, 60]]],dtype=np.float32)
        >>> y = C.constant(value=data2)
        >>> # splice both inputs on axis=0 returns a 5x2 matrix
        >>> C.splice(x, y, axis=1).eval()
        array([[[  1.,   2.],
                [  4.,   5.],
                [ 10.,  20.],
                [ 30.,  40.],
                [ 50.,  60.]]], dtype=float32)

    Args:
        inputs: one or more input tensors
        axis (int or :class:`~cntk.axis.Axis`, optional, keyword only): axis along which the
         concatenation will be performed
        name (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    axis, name = (lambda axis=-1, name='': (axis, name))(**kw_axis_name) # Python 2.7 does not allow (*inputs, axis=-1, name='')
    # short-circuit if nothing to splice
    if len(inputs) == 1:
        return combine([inputs[0]]) # (but make it into a Function)

    from cntk.cntk_py import splice

    inputs = [sanitize_input(x) for x in inputs]
    axis = sanitize_axis(axis)

    return splice(inputs, axis, name) # C++ projection expects inputs as a list

@typemap
def unpack_batch(x, name=''):
    '''
    Concatenate the input tensor's last dynamic axis to static axis.
    Only tensors with batch axis are supported now.

    Example:
        >>> data = np.arange(12).reshape((3,2,2))
        >>> x = C.input((2,2))
        >>> C.unpack_batch(x).eval({x:data})
        array([[[  0.,   1.],
                [  2.,   3.]],
        <BLANKLINE>
               [[  4.,   5.],
                [  6.,   7.]],
        <BLANKLINE>
               [[  8.,   9.],
                [ 10.,  11.]]], dtype=float32)

    Args:
        x: a tensor with dynamic axis
        name: (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import unpack_batch
    x = sanitize_input(x)
    return unpack_batch(x, name)

@typemap
def to_batch(x, name=''):
    '''
    Concatenate the input tensor's first axis to batch axis.

    Example:
        >>> data = np.arange(12).reshape((3,2,2))
        >>> x = C.constant(value=data)
        >>> y = C.to_batch(x)
        >>> y.shape
        (2, 2)

    Args:
        x: a tensor with dynamic axis
        name: (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import to_batch
    x = sanitize_input(x)
    return to_batch(x, name)

@typemap
def one_hot(x, num_classes, sparse_output=False, axis=-1, name=''):
    '''
    Create one hot tensor based on the input tensor

    Example:
        >>> data = np.asarray([[1, 2],
        ...                    [4, 5]], dtype=np.float32)

        >>> x = C.input_variable((2,))
        >>> C.one_hot(x, 6, False).eval({x:data})
        array([[[ 0.,  1.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  1.,  0.,  0.,  0.]],
        <BLANKLINE>
                [[ 0.,  0.,  0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  0.,  0.,  1.]]], dtype=float32)

    Args:
        x: input tensor, the value must be positive integer and less than num_class
        num_classes: the number of class in one hot tensor
        sparse_output: if set as True, we will create the one hot tensor as sparse.
        axis: The axis to fill (default: -1, a new inner-most axis).
        name (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import one_hot_op
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return one_hot_op(x, num_classes, sparse_output, axis, name)

@typemap
def gather(reference, indices, axis=None, name=''):
    '''
    Retrieves the elements of indices in the tensor reference.

    Example:
        >>> c = np.asarray([[[0],[1]],[[4],[5]]]).astype('f')
        >>> x = C.input_variable((2,1))
        >>> d = np.arange(12).reshape(6,2).astype('f')
        >>> y = C.constant(d)
        >>> C.gather(y, x).eval({x:c})
        array([[[[  0.,   1.]],
        <BLANKLINE>
                [[  2.,   3.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[  8.,   9.]],
        <BLANKLINE>
                [[ 10.,  11.]]]], dtype=float32)

    Args:
        reference: A tensor of values
        indices: A tensor of indices
        axis: The axis along which the indices refer to. Default (None) means the  first axis. Only one axis is supported;
        and only static axis is supported.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import gather_op
    dtype = get_data_type(reference)
    indices = sanitize_input(indices, dtype)
    reference = sanitize_input(reference, dtype)

    if axis is None:
        return gather_op(indices, reference, name)
    else:
        axis = sanitize_axis(axis)
        return gather_op(indices, reference, axis, name)

@typemap
def flatten(x, axis = None, name = ''):
    '''
    Flattens the input tensor into a 2D matrix.
    If the input tensor has shape (d_0, d_1, ... d_n) then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).

    Example:
        >>> # create 2x3x4 matrix, flatten the matrix at axis = 1
        >>> shape = (2, 3, 4) 
        >>> data = np.reshape(np.arange(np.prod(shape), dtype = np.float32), shape)
        >>> C.flatten(data, 1).eval()
        array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
                 11.],
               [ 12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,
                 23.]], dtype=float32)

    Args:
        x: Input tensor.
        axis (int): (Default to 0) Indicates up to which input dimensions (exclusive) should be flattened to the outer dimension of the output
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import flatten
    x = sanitize_input(x)
    if axis is not None:
        axis = sanitize_axis(axis)
        return flatten(x, axis, name)
    else:
        return flatten(x, name)

##########################################################################
# reduction ops
##########################################################################

@typemap
def reduce_sum(x, axis=None, keepdims=True, name=''):
    '''
    Computes the sum of the input tensor's elements across one axis or a list of axes. If the axis parameter
    is not specified then the sum will be computed over all static axes, which is
    equivalent with specifying ``axis=Axis.all_static_axes()``. If
    ``axis=Axis.all_axes()`` is specified, then the output is a scalar which is the sum of all the
    elements in the minibatch. And if ``axis=Axis.default_batch_axis()`` is specified, then the reduction
    will happen across the batch axis (In this case the input must not be a sequence).

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_sum(data, 0).eval().round(4)
        array([[[  90.,    3.],
                [ 120.,    6.]]], dtype=float32)
        >>> np.sum(data, axis=0).round(4)
        array([[  90.,    3.],
               [ 120.,    6.]], dtype=float32)
        >>> C.reduce_sum(data, 1).eval().round(4)
        array([[[  25.,    3.]],
        <BLANKLINE>
               [[  70.,    3.]],
        <BLANKLINE>
               [[ 115.,    3.]]], dtype=float32)
        >>> np.sum(data, axis=1).round(4)
        array([[  25.,    3.],
               [  70.,    3.],
               [ 115.,    3.]], dtype=float32)
        >>> C.reduce_sum(data, (0,2)).eval().round(4)
        array([[[  93.],
                [ 126.]]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    from cntk.cntk_py import reduce_sum
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_sum(x, axis, keepdims, name)


@typemap
def reduce_log_sum_exp(x, axis=None, keepdims=True, name=''):
    '''
    Computes the log of the sum of the exponentiations of the input tensor's
    elements across a specified axis or a list of specified axes.

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_log_sum_exp(data, axis=0).eval().round(4)
        array([[[ 55.      ,   2.0986],
                [ 60.      ,   3.0986]]], dtype=float32)
        >>> np.log(np.sum(np.exp(data), axis=0)).round(4)
        array([[ 55.      ,   2.0986],
               [ 60.      ,   3.0986]], dtype=float32)
        >>> C.reduce_log_sum_exp(data, axis=(0,2)).eval().round(4)
        array([[[ 55.],
                [ 60.]]], dtype=float32)
        >>> np.log(np.sum(np.exp(data), axis=(0,2))).round(4)
        array([ 55.,  60.], dtype=float32)

        >>> x = C.input_variable(shape=(2,2))
        >>> lse = C.reduce_log_sum_exp(x, axis=[C.axis.Axis.default_batch_axis(), 1])
        >>> lse.eval({x:data}).round(4)
        array([[ 55.],
               [ 60.]], dtype=float32)
        >>> np.log(np.sum(np.exp(data), axis=(0,2))).round(4)
        array([ 55.,  60.], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    # TODO: rename V2 API function as well from reduce_log_sum() to reduce_log_sum_exp()
    from cntk.cntk_py import reduce_log_sum
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_log_sum(x, axis, keepdims, name)


@typemap
def reduce_mean(x, axis=None, keepdims=True, name=''):
    '''
    Computes the mean of the input tensor's elements across a specified axis or a list of specified axes.

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_mean(data, 0).eval().round(4)
        array([[[ 30.,   1.],
                [ 40.,   2.]]], dtype=float32)
        >>> np.mean(data, axis=0).round(4)
        array([[ 30.,   1.],
               [ 40.,   2.]], dtype=float32)
        >>> C.reduce_mean(data, 1).eval().round(4)
        array([[[ 12.5,   1.5]],
        <BLANKLINE>
               [[ 35. ,   1.5]],
        <BLANKLINE>
               [[ 57.5,   1.5]]], dtype=float32)
        >>> np.mean(data, axis=1).round(4)
        array([[ 12.5,   1.5],
               [ 35. ,   1.5],
               [ 57.5,   1.5]], dtype=float32)
        >>> C.reduce_mean(data, (0,2)).eval().round(4)
        array([[[ 15.5],
                [ 21. ]]], dtype=float32)

        >>> x = C.input_variable((2,2))
        >>> C.reduce_mean( x * 1.0, (C.Axis.default_batch_axis(), 1)).eval({x: data}).round(4)
        array([[ 15.5],
               [ 21.      ]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

     Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    from cntk.cntk_py import reduce_mean
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_mean(x, axis, keepdims, name)


@typemap
def reduce_max(x, axis=None, keepdims=True, name=''):
    '''
    Computes the max of the input tensor's elements across a specified axis or a list of specified axes.

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_max(data, 0).eval().round(4)
        array([[[ 55.,   1.],
                [ 60.,   2.]]], dtype=float32)
        >>> C.reduce_max(data, 1).eval().round(4)
        array([[[ 20.,   2.]],
        <BLANKLINE>
               [[ 40.,   2.]],
        <BLANKLINE>
               [[ 60.,   2.]]], dtype=float32)
        >>> C.reduce_max(data, (0,2)).eval().round(4)
        array([[[ 55.],
                [ 60.]]], dtype=float32)

        >>> x = C.input_variable((2,2))
        >>> C.reduce_max( x * 1.0, (C.Axis.default_batch_axis(), 1)).eval({x: data}).round(4)
        array([[ 55.],
               [ 60.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    from cntk.cntk_py import reduce_max
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_max(x, axis, keepdims, name)


@typemap
def reduce_min(x, axis=None, keepdims=True, name=''):
    '''
    Computes the min of the input tensor's elements across a specified axis or a list of specified axes.

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_min(data, 0).eval().round(4)
        array([[[  5.,   1.],
                [ 20.,   2.]]], dtype=float32)
        >>> C.reduce_min(data, 1).eval().round(4)
        array([[[  5.,   1.]],
        <BLANKLINE>
               [[ 30.,   1.]],
        <BLANKLINE>
               [[ 55.,   1.]]], dtype=float32)
        >>> C.reduce_min(data, (0,2)).eval().round(4)
        array([[[ 1.],
                [ 2.]]], dtype=float32)

        >>> x = C.input_variable((2,2))
        >>> C.reduce_min( x * 1.0, (C.Axis.default_batch_axis(), 1)).eval({x: data}).round(4)
        array([[ 1.],
               [ 2.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a list of integers or a list of :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

     Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    from cntk.cntk_py import reduce_min
    x = sanitize_input(x)
    axis = sanitize_axis_list(axis)
    return reduce_min(x, axis, keepdims, name)


@typemap
def reduce_prod(x, axis=None, keepdims=True, name=''):
    '''
    Computes the min of the input tensor's elements across the specified axis.

    Example:
        >>> # create 3x2x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32)

        >>> C.reduce_prod(data, 0).eval().round(4)
        array([[[  8250.,      1.],
                [ 48000.,      8.]]], dtype=float32)
        >>> C.reduce_prod(data, 1).eval().round(4)
        array([[[  100.,     2.]],
        <BLANKLINE>
               [[ 1200.,     2.]],
        <BLANKLINE>
               [[ 3300.,     2.]]], dtype=float32)
        >>> C.reduce_prod(data, (0,2)).eval().round(4)
        array([[[   8250.],
                [ 384000.]]], dtype=float32)

        >>> x = C.input_variable((2,2))
        >>> C.reduce_prod( x * 1.0, (C.Axis.default_batch_axis(), 1)).eval({x: data}).round(4)
        array([[   8250.],
               [ 384000.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Note that CNTK keeps the shape of the resulting tensors when reducing over multiple static axes.
    '''
    from cntk.cntk_py import reduce_prod
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_prod(x, axis, keepdims, name)

@typemap
def reduce_l1(x, axis=None, keepdims = True, name=''):
    '''
    Computes the L1 norm of the input tensor's element along the provided axes. 
    The resulted tensor has the same rank as the input if keepdims equal 1. 
    If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.

    Example:
        >>> C.reduce_l1([[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], 2, False).eval()
        array([[  3.,   7.],
               [ 11.,  15.],
               [ 19.,  23.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_l1
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_l1(x, axis, keepdims, name)

@typemap
def reduce_l2(x, axis=None, keepdims = True, name=''):
    '''
    Computes the L2 norm of the input tensor's element along the provided axes. 
    The resulted tensor has the same rank as the input if keepdims equal 1. 
    If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.

    Example:
        >>> C.reduce_l2([[[1,2], [3,4]]], 2).eval()
        array([[[ 2.236068],
                [ 5.        ]]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_l2
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_l2(x, axis, keepdims, name)

@typemap
def reduce_sum_square(x, axis=None, keepdims = True, name=''):
    '''
    Computes the sum square of the input tensor's element along the provided axes. 
    The resulted tensor has the same rank as the input if keepdims equal 1. 
    If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.

    Example:
        >>> C.reduce_sum_square([[[1,2], [3,4]]], 2).eval()
        array([[[  5.],
                [ 25.]]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis` or a :obj:`list` or :obj:`tuple` of int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        keepdims (boolean): Keep the reduced dimension or not, default True mean keep reduced dimension
        name (str): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_sum_square
    x = sanitize_input(x)
    axis = sanitize_multi_axis_reduction_list(axis)
    return reduce_sum_square(x, axis, keepdims, name)

@typemap
def image_scaler(x, scalar, biases, name=''):
    '''
    Alteration of image by scaling its individual values.

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        scalar (float): Scalar channel factor.
        bias (numpy array): Bias values for each channel.

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import image_scaler
    x = sanitize_input(x)
    return image_scaler(x, scalar, biases, name)

@typemap
def argmax(x, axis=None, name=''):
    '''
    Computes the argmax of the input tensor's elements across the specified axis.
    If no axis is specified, it will return the flatten index of the largest element
    in tensor x.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 20],[30, 40],[50, 60]]

        >>> C.argmax(data, 0).eval()
        array([[ 2.,  2.]], dtype=float32)

        >>> C.argmax(data, 1).eval()
        array([[ 1.],
               [ 1.],
               [ 1.]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import argmax
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return argmax(x, axis, name)

@typemap
def argmin(x, axis=None, name=''):
    '''
    Computes the argmin of the input tensor's elements across the specified axis.
    If no axis is specified, it will return the flatten index of the smallest element
    in tensor x.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 30],[40, 20],[60, 50]]

        >>> C.argmin(data, 0).eval()
        array([[ 0.,  1.]], dtype=float32)

        >>> C.argmin(data, 1).eval()
        array([[ 0.],
               [ 1.],
               [ 1.]], dtype=float32)

    Args:
        x (`numpy.array` or :class:`~cntk.ops.functions.Function`): any :class:`~cntk.ops.functions.Function` that outputs a tensor.
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str, default to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        An instance of :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import argmin
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return argmin(x, axis, name)

@typemap
def to_sequence(x, sequence_lengths=None, sequence_axis_name_prefix='toSequence_', name=''):
    '''
    This function converts 'x' to a sequence using the most significant
    static axis [0] as the sequence axis.

    The sequenceLengths input is optional; if unspecified, all sequences are
    assumed to be of the same length; i.e. dimensionality of the most significant
    static axis

    Args:
        x: the tensor (or its name) which is converted to a sequence
        sequence_lengths: Optional tensor operand representing the sequence lengths.
            if unspecified, all sequences are assumed to be of the same length;
            i.e. dimensionality of the most significant static axis.
        sequence_axis_name_prefix (str, optional): prefix of the new sequence axis name.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Todo:
        add an example
    '''

    from cntk.cntk_py import to_sequence

    x = sanitize_input(x)
    if sequence_lengths is None:
        return to_sequence(x, sequence_axis_name_prefix, name)
    else:
        sequence_lengths = sanitize_input(sequence_lengths)
        return to_sequence(x, sequence_lengths, sequence_axis_name_prefix, name)


@typemap
def to_sequence_like(x, dynamic_axes_like, name=''):
    '''
    This function converts 'x' to a sequence using the most significant
    static axis [0] as the sequence axis. The length of the sequences are
    obtained from the 'dynamic_axes_like' operand.

    Args:
        x: the tensor (or its name) which is converted to a sequence
        dynamic_axes_like: Tensor operand used to obtain the lengths of
            the generated sequences. The dynamic axes of the generated sequence
            tensor match the dynamic axes of the 'dynamic_axes_like' operand.
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Todo:
        add an example
    '''

    from cntk.cntk_py import to_sequence_like

    x = sanitize_input(x)
    dynamic_axes_like = sanitize_input(dynamic_axes_like)
    return to_sequence_like(x, dynamic_axes_like, name)

#######################################################################
# training ops
#######################################################################

@typemap
def random_sample(
    weights,
    num_samples,
    allow_duplicates,
    seed = SentinelValueForAutoSelectRandomSeed,
    name=''):
    '''
    Estimates inclusion frequencies for random sampling with or without
    replacement.

    The output value is a set of num_samples random samples represented
    by a (sparse) matrix of shape [num_samples x len(weights)],
    where len(weights) is the number of classes (categories) to choose
    from. The output has no dynamic axis.
    The samples are drawn according to the weight vector p(i) =
    weights[i] / sum(weights)
    We get one set of samples per minibatch.
    Intended use cases are e.g. sampled softmax, noise contrastive
    estimation etc.

    Args:
        weights: input vector of sampling weights which should be
            non-negative numbers.
        num_samples (int): number of expected samples
        allow_duplicates (bool): If sampling is done
            with replacement (`True`) or without (`False`).
        seed (int): random seed.
        name (:class:`str`, optional): the name of the Function instance in the network.

    Returns:
        :class:`~cntk.ops.functions.Function`

    '''
    from cntk.cntk_py import random_sample
    weights = sanitize_input(weights)

    return random_sample(weights, num_samples, allow_duplicates, seed, name)


@typemap
def random_sample_inclusion_frequency(
    weights,
    num_samples,
    allow_duplicates,
    seed = SentinelValueForAutoSelectRandomSeed,
    name=''):
    '''
    For weighted sampling with the specified sample size (`num_samples`)
    this operation computes the expected number of occurrences of each class
    in the sampled set. In case of sampling without replacement
    the result is only an estimate which might be quite rough in the
    case of small sample sizes.
    Intended uses are e.g. sampled softmax, noise contrastive
    estimation etc.
    This operation will be typically used together
    with :func:`random_sample`.

    Args:
        weights: input vector of sampling weights which should be
         non-negative numbers.
        num_samples (int): number of expected samples
        allow_duplicates (bool): If sampling is done
         with replacement (`True`) or without (`False`).
        seed (int): random seed.
        name (:class:`str`, optional): the name of the Function instance in the network.

    Example:
        >>> import numpy as np
        >>> from cntk import *
        >>> # weight vector with 100 '1000'-values followed
        >>> # by 100 '1' values
        >>> w1 = np.full((100),1000, dtype = np.float)
        >>> w2 = np.full((100),1, dtype = np.float)
        >>> w = np.concatenate((w1, w2))
        >>> f = random_sample_inclusion_frequency(w, 150, True).eval()
        >>> f[0]
        1.4985015
        >>> f[1]
        1.4985015
        >>> f[110]
        0.0014985015
        >>> # when switching to sampling without duplicates samples are
        >>> # forced to pick the low weight classes too
        >>> f = random_sample_inclusion_frequency(w, 150, False).eval()
        >>> f[0]
        1.0

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import random_sample_inclusion_frequency
    weights = sanitize_input(weights)

    return random_sample_inclusion_frequency(
        weights,
        num_samples,
        allow_duplicates,
        seed,
        name)


@typemap
def dropout(x, dropout_rate=0.0, seed = SentinelValueForAutoSelectRandomSeed, name=''):
    '''
    Each element of the input is independently set to 0 with probability ``dropout_rate``
    or to 1 / (1 - ``dropout_rate``) times its original value (with probability 1-``dropout_rate``).
    Dropout is a good way to reduce overfitting.

    This behavior only happens during training. During inference dropout is a no-op.
    In the paper that introduced dropout it was suggested to scale the weights during inference.
    In CNTK's implementation, because the values that are not set to 0 are multiplied
    with (1 / (1 - ``dropout_rate``)), this is not necessary.

    Example:
        >>> data = [[10, 20],[30, 40],[50, 60]]
        >>> C.dropout(data, 0.5).eval() # doctest: +SKIP
        array([[  0.,  40.],
               [  0.,  80.],
               [  0.,   0.]], dtype=float32)

        >>> C.dropout(data, 0.75).eval() # doctest: +SKIP
        array([[   0.,    0.],
               [   0.,  160.],
               [   0.,  240.]], dtype=float32)

    Args:
        x: input tensor
        dropout_rate (float, [0,1)): probability that an element of ``x`` will be set to zero
        seed (int): random seed.
        name (:class:`str`, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        raise ValueError('dropout_rate must be in the interval [0,1)')

    from cntk.cntk_py import dropout
    x = sanitize_input(x)

    return dropout(x, dropout_rate, seed, name)

##########################################################################
# variables_and_parameters ops
##########################################################################

from cntk.device import use_default_device
from cntk.axis import Axis

def _input_spec(shape, dtype=default_override_or(np.float32), needs_gradient=False, is_sparse=False,
                dynamic_axes=[Axis.default_batch_axis()], name=''):
    '''
    We need _input_spec because input is python built-in and because of typemap, must remain
    in sync with input.
    TODO: Investigate to remove it.
    '''
    pass

@typemap
def input(shape, dtype=default_override_or(np.float32), needs_gradient=False, is_sparse=False,
          dynamic_axes=[Axis.default_batch_axis()], name=''):
    '''
    DEPRECATED.

    It creates an input in the network: a place where data,
    such as features and labels, should be provided.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        needs_gradient (bool, optional): whether to back-propagates to it or not. False by default.
        is_sparse (bool, optional): whether the variable is sparse (`False` by default)
        dynamic_axes (list or tuple, default): a list of dynamic axis (e.g., batch axis, sequence axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.variables.Variable`
    '''
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
                  'input_variable() instead.', DeprecationWarning)

    return input_variable(shape, dtype, needs_gradient, is_sparse, dynamic_axes, name)

@typemap
def input_variable(shape, dtype=default_override_or(np.float32), needs_gradient=False, is_sparse=False,
                   dynamic_axes=[Axis.default_batch_axis()], name=''):
    '''
    input_variable(shape, dtype=np.float32, needs_gradient=False, is_sparse=False, dynamic_axes=[Axis.default_batch_axis()], name='')

    It creates an input in the network: a place where data,
    such as features and labels, should be provided.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (np.float32 or np.float64 or np.float16): data type. Default is np.float32.
        needs_gradient (bool, optional): whether to back-propagates to it or not. False by default.
        is_sparse (bool, optional): whether the variable is sparse (`False` by default)
        dynamic_axes (list or tuple, default): a list of dynamic axis (e.g., batch axis, time axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.variables.Variable`
    '''
    from cntk.cntk_py import input_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk

    shape = sanitize_shape(shape)
    dtype = get_default_override(_input_spec, dtype=dtype)
    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)

    # TODO dynamic axis for numpy arrays
    # TODO sparse for numpy arrays

    return input_variable(shape, is_sparse, dtype, needs_gradient, name, dynamic_axes)

@typemap
def output_variable(shape, dtype, dynamic_axes, needs_gradient=True, name=''):
    '''
    It creates an output variable that is used to define a user defined function.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (np.float32 or np.float64 or np.float16): data type
        dynamic_axes (list or tuple): a list of dynamic axis (e.g., batch axis, time axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.variables.Variable` that is of output type
    '''
    from cntk.cntk_py import output_variable
    from cntk.internal import sanitize_shape, sanitize_dtype_cntk

    shape = sanitize_shape(shape)

    dtype = sanitize_dtype_cntk(dtype)

    for a in dynamic_axes:
        if not a.is_dynamic_axis:
            raise ValueError('axis in dynamic_axes attribute is not dynamic')
    dynamic_axes = list(reversed(dynamic_axes))

    return output_variable(shape, dtype, dynamic_axes, needs_gradient, name)

@typemap
def placeholder(shape=None, dynamic_axes=None, name=''):
    '''
    It creates a placeholder variable that has to be later bound to an actual variable.
    A common use of this is to serve as a placeholder for a later output variable in a
    recurrent network, which is replaced with the actual output variable by calling
    replace_placeholder(s).

    Args:
        shape (tuple or int): the shape of the variable tensor
        dynamic_axes (list): the list of dynamic axes that the actual variable uses
        name (str, optional): the name of the placeholder variable in the network

    Returns:
        :class:`~cntk.variables.Variable`
    '''
    from cntk.cntk_py import placeholder_variable, NDShape, Axis

    if shape is None:
        shape = NDShape.unknown().dimensions()
    else:
        shape = sanitize_shape(shape)

    if dynamic_axes is None:
        dynamic_axes = Axis.unknown_dynamic_axes()

    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)
    return placeholder_variable(shape, name, dynamic_axes)

@typemap
def parameter(shape=None, init=None, dtype=None, device=None, name=''):
    '''
    It creates a parameter tensor.

    Example:
        >>> init_parameter = C.parameter(shape=(3,4), init=2)
        >>> np.asarray(init_parameter) # doctest: +SKIP
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)

    Args:
        shape (tuple or int, optional): the shape of the input tensor. If not provided, it
         will be inferred from ``value``.
        init (scalar or NumPy array or initializer): if init is a scalar
         it will be replicated for every element in the tensor or
         NumPy array. If it is the output of an initializer form
         :mod:`cntk.initializer` it will be used to initialize the tensor at
         the first forward pass. If `None`, the tensor will be initialized
         with 0.
        dtype (optional): data type of the constant. If a NumPy array and ``dtype``,
         are given, then data will be converted if needed. If none given, it will default to ``np.float32``.
        device (:class:`~cntk.device.DeviceDescriptor`): instance of DeviceDescriptor
        name (str, optional): the name of the Parameter instance in the network

    Returns:
        :class:`~cntk.variables.Parameter`
    '''

    from ..variables import Parameter
    return Parameter(shape, init, dtype, device, name)


@typemap
def constant(value=None, shape=None, dtype=None, device=None, name=''):
    '''
    It creates a constant tensor initialized from a numpy array.

    Example:
        >>> constant_data = C.constant([[1., 2.], [3., 4.], [5., 6.]])
        >>> constant_data.value
        array([[ 1.,  2.],
               [ 3.,  4.],
               [ 5.,  6.]], dtype=float32)

    Args:
        value (scalar or NumPy array, optional): a scalar initial value that would be replicated for
         every element in the tensor or NumPy array.
         If ``None``, the tensor will be initialized uniformly random.
        shape (tuple or int, optional): the shape of the input tensor. If not provided, it will
         be inferred from ``value``.
        dtype (optional): data type of the constant. If a NumPy array and ``dtype``,
         are given, then data will be converted if needed. If none given, it will default to ``np.float32``.
        device (:class:`~cntk.device.DeviceDescriptor`): instance of DeviceDescriptor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.variables.Constant`
    '''
    from ..variables import Constant
    return Constant(value, shape, dtype, device, name)

##########################################################################
# normalization ops
##########################################################################

# TODO: ComputeInputPerDimMeansAndInvStdDevs


@typemap
def per_dim_mean_variance_normalize(operand, mean, inv_stddev, name=''):
    '''
    Computes per dimension mean-variance normalization of the specified input operand.

    Args:
        operand: the variable to be normalized
        mean (NumPy array): per dimension mean to use for the normalization
        inv_stddev (NumPy array): per dimension standard deviation to use for the normalization
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import per_dim_mean_variance_normalize
    mean = sanitize_input(mean, get_data_type(mean))
    inv_stddev = sanitize_input(inv_stddev, get_data_type(inv_stddev))
    return per_dim_mean_variance_normalize(operand, mean, inv_stddev, name)

@typemap
def stop_gradient(input, name=''):
    '''
    Outputs its input as it is and prevents any gradient contribution from its output to its input.

    Args:
        input: class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import stop_gradient
    dtype = get_data_type(input)
    op = sanitize_input(input, dtype)
    return stop_gradient(op, name)

@typemap
def assign(ref, input, name=''):
    '''
    Assign the value in input to ref and return the new value, ref need to be the same layout as input.
    Both ref and input can't have dynamic axis and broadcast isn't supported for the assign operator.
    During forward pass, ref will get the new value after the forward or backward pass finish, so that
    any part of the graph that depend on ref will get the old value. To get the new value, use the one
    returned by the assign node. The reason for that is to make ``assign`` have a deterministic behavior.

    If not computing gradients, the ref will be assigned the new value after the forward pass over the
    entire Function graph is complete; i.e. all uses of ref in the forward pass will use the original
    (pre-assignment) value of ref.

    If computing gradients (training mode), the assignment to ref will happen after completing both
    the forward and backward passes over the entire Function graph.

    The ref must be a Parameter or Constant. If the same ref is used in multiple assign operations,
    then the order in which the assignment happens is non-deterministic and the final value can be
    either of the assignments unless an order is established using a data dependence between the
    assignments.

    Example:
        >>> dest = C.constant(shape=(3,4))
        >>> data = C.parameter(shape=(3,4), init=2)
        >>> C.assign(dest,data).eval()
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)
        >>> dest.asarray()
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)

        >>> dest = C.parameter(shape=(3,4), init=0)
        >>> a = C.assign(dest, data)
        >>> y = dest + data
        >>> result = C.combine([y, a]).eval()
        >>> result[y.output]
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)
        >>> dest.asarray()
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)
        >>> result = C.combine([y, a]).eval()
        >>> result[y.output]
        array([[ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.]], dtype=float32)
        >>> dest.asarray()
        array([[ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  2.]], dtype=float32)

    Args:
        ref: class: `~cntk.variables.Constant` or `~cntk.variables.Parameter`.
        input: class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import assign
    dtype = get_data_type(input)
    operand = sanitize_input(input, dtype)
    ref_operand = sanitize_input(ref, dtype)
    return assign(ref_operand, operand, name)

@typemap
def crop_manual(node_input, node_referent, offset_x, offset_y, name = ''):
    '''
    Crops input along spatial dimensions so that it matches spatial size of reference input.
    Crop offsets are given in pixels.

    Args:
        node_input: class:`~cntk.ops.functions.Function` that outputs the tensor to be cropped
        node_referent: class:`~cntk.ops.functions.Function` that outputs the reference tensor
        offset_x (int): horizontal crop offset
        offset_y (int): vertical crop offset
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import crop
    arg_input = sanitize_input(node_input, get_data_type(node_input))
    arg_ref = sanitize_input(node_referent,  get_data_type(node_referent))
    return crop(arg_input, arg_ref, offset_x, offset_y, name)

@typemap
def crop_automatic(node_input, node_referent, name = ''):
    '''
    Crops input along spatial dimensions so that it matches spatial size of reference input.

    Crop offsets are computed by traversing the network graph and computing affine transform
    between the two inputs. Translation part of the transform determines the offsets. The transform
    is computed as composition of the transforms between each input and their common ancestor.
    The common ancestor is expected to exist.

    Args:
        node_input: class:`~cntk.ops.functions.Function` that outputs the tensor to be cropped
        node_referent: class:`~cntk.ops.functions.Function` that outputs the reference tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import crop
    arg_input = sanitize_input(node_input, get_data_type(node_input))
    arg_ref = sanitize_input(node_referent,  get_data_type(node_referent))
    return crop(arg_input, arg_ref, name)

@typemap
def crop_automatic_with_ancestors(node_input, node_referent, ancestor_input, ancestor_referent, name = ''):
    '''
    Crops input along spatial dimensions so that it matches spatial size of reference input.

    Crop offsets are computed by traversing the network graph and computing affine transform
    between the two inputs. Translation part of the transform determines the offsets. The transform
    is computed as composition of the transforms between each input and their common ancestor.

    ancestor_input and ancestor_referent are expected to be ancestors of node_input and
    node_referent, respectively. They act like the same node for the purpose of finding a common
    ancestor. They are used in cases when node_input and node_referent do not have a common
    ancestor in the network. Typically, the ancestor nodes have the same spatial size. For example, in
    pixelwise semantic labeling, ancestor_input would be the input image, and ancestor_referent would
    be the ground truth image containing pixelwise labels.

    Args:
        node_input: class:`~cntk.ops.functions.Function` that outputs the tensor to be cropped
        node_referent: class:`~cntk.ops.functions.Function` that outputs the reference tensor
        ancestor_input (optional): class:`~cntk.ops.functions.Function` that outputs ancestor of node_input
        ancestor_referent (optional): class:`~cntk.ops.functions.Function` that outputs ancestor of node_referent
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import crop
    arg_node_input = sanitize_input(node_input, get_data_type(node_input))
    arg_node_ref = sanitize_input(node_referent,  get_data_type(node_referent))
    arg_ancestor_input = sanitize_input(ancestor_input, get_data_type(ancestor_input))
    arg_ancestor_ref = sanitize_input(ancestor_referent,  get_data_type(ancestor_referent))
    return crop(arg_node_input, arg_node_ref, arg_ancestor_input, arg_ancestor_ref, name)

@typemap
def depth_to_space(operand, block_size, name=''):
    '''
    Rearranges elements in the input tensor from the depth dimension into spatial blocks.

    This operation is useful for implementing sub-pixel convolution that is part of models  
    for image super-resolution (see [1]). It rearranges elements of an input tensor of shape 
    (Cxbxb, H, W) to a tensor of shape (C, bxH, bxW), where b is the `block_size`.
    
    Example:
        >>> x = np.array(np.reshape(range(8), (8, 1, 1)), dtype=np.float32)
        >>> x = np.tile(x, (1, 2, 3))
        >>> a = C.input_variable((8, 2, 3))
        >>> d2s_op = C.depth_to_space(a, block_size=2)
        >>> d2s_op.eval({a:x})
        array([[[[ 0.,  2.,  0.,  2.,  0.,  2.],
                 [ 4.,  6.,  4.,  6.,  4.,  6.],
                 [ 0.,  2.,  0.,  2.,  0.,  2.],
                 [ 4.,  6.,  4.,  6.,  4.,  6.]],
        <BLANKLINE>
                [[ 1.,  3.,  1.,  3.,  1.,  3.],
                 [ 5.,  7.,  5.,  7.,  5.,  7.],
                 [ 1.,  3.,  1.,  3.,  1.,  3.],
                 [ 5.,  7.,  5.,  7.,  5.,  7.]]]], dtype=float32)

    Args:
        operand: Input tensor, with dimensions :math:`[C \\times H \\times W]`.
        block_size (int): Integer value. This defines the size of the spatial block where the 
         depth elements move to. Number of channels, C, in the input tensor must be divisible 
         by math:`(block_size \\times block_size)`
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        [1] W. Shi et. al. `: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_.
    '''
    from cntk.cntk_py import depth_to_space
    operand = sanitize_input(operand, get_data_type(operand))
    if not float(block_size).is_integer():
        raise ValueError('block_size must be an integer.')
    return depth_to_space(operand, block_size, name)

@typemap
def space_to_depth(operand, block_size, name=''):
    '''
    Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.

    This is the reverse transformation of depth_to_space. This operation is useful for implementing 
    and testing sub-pixel convolution that is part of models for image super-resolution (see [1]).
    It rearranges elements of an input tensor of shape (C, H, W) to a tensor of shape (C*b*b, H/b, W/b),
    where b is the `block_size`, by rearranging non-overlapping spatial blocks of size `block_size` x `block_size`
    into the depth/channel dimension at each location.
    
    Example:
        >>> np.random.seed(3)
        >>> x = np.random.randint(low=0, high=100, size=(1, 4, 6)).astype(np.float32)
        >>> a = C.input_variable((1, 4, 6))
        >>> s2d_op = C.space_to_depth(a, block_size=2)
        >>> s2d_op.eval({a:x})
        array([[[[ 24.,  56.,   0.],
                 [ 96.,  44.,  39.]],
        <BLANKLINE>
                [[  3.,  72.,  21.],
                 [ 20.,  93.,  14.]],
        <BLANKLINE>
                [[ 19.,  41.,  21.],
                 [ 26.,  90.,  66.]],
        <BLANKLINE>
                [[ 74.,  10.,  38.],
                 [ 81.,  22.,   2.]]]], dtype=float32)

    Args:
        operand: Input tensor, with dimensions :math:`[C \\times H \\times W]`.
        block_size (int): Integer value. This defines the size of the spatial block whose elements
         are moved to the depth dimension. Size of spatial dimensions (H, W) in the input tensor
         must be divisible by math:`block_size`
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        [1] W. Shi et. al. `: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_.
    '''
    from cntk.cntk_py import space_to_depth
    operand = sanitize_input(operand, get_data_type(operand))
    if not float(block_size).is_integer():
        raise ValueError('block_size must be an integer.')
    return space_to_depth(operand, block_size, name)

@typemap
def cast(node_input, dtype, name=''):
    '''
    cast input to dtype, with the same shape and dynamic axes. This function currently only support forward.
    
    Args:
        node_input: class:`~cntk.input_variable` that needs the dtype conversion
        dtype: data_type (np.float32, np.float64, np.float16): data type of the converted output
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`    
    '''
    from cntk.cntk_py import cast
    arg_node_input = sanitize_input(node_input, get_data_type(node_input))
    arg_dtype = sanitize_dtype_cntk(dtype)
    return cast(arg_node_input, arg_dtype, name)

@typemap
def mean_variance_normalization(operand, epsilon=0.00001, use_stats_across_channels=False, do_variance_scaling=True, name=''):
    '''
    Computes mean-variance normalization of the specified input operand. 

    This operation computes and mean and variance for the entire tensor if use_stats_across_channels is True. 
    If use_stats_across_channels is False the computes mean and variance per channel and normalizes each 
    channel with its own mean and variance. If do_variance_scaling is False, only the mean is subtracted,
    and the variance scaling is omitted.
    
    Example:
        >>> data = np.array([[[0., 2], [4., 6.]], [[0., 4], [8., 12.]]]).astype(np.float32)
        >>> data
        array([[[  0.,   2.],
                [  4.,   6.]],
        <BLANKLINE>
               [[  0.,   4.],
                [  8.,  12.]]], dtype=float32)
        >>> saved_precision = np.get_printoptions()['precision']
        >>> np.set_printoptions(precision=4) # For consistent display upto 4 decimals.
        >>> C.mean_variance_normalization(data).eval()
        array([[[-1.3416, -0.4472],
                [ 0.4472,  1.3416]],
        <BLANKLINE>
               [[-1.3416, -0.4472],
                [ 0.4472,  1.3416]]], dtype=float32)
        >>> np.set_printoptions(precision=saved_precision) # Reseting the display precision.

    Args:
        operand: Input tensor, with dimensions :math:`[C \\times H \\times W]`.
        epsilon (double, default 0.00001): epsilon added to the standard deviation to avoid division by 0.
        use_stats_across_channels (bool): If False, mean and variance are computed per channel.
         If True, mean and variance are computed over the entire tensor (all axes).
        do_variance_scaling (bool): If False, only the mean is subtracted. If True, it is also
         scaled by inverse of standard deviation.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    
    '''
    from cntk.cntk_py import mean_variance_normalization
    if epsilon < 0:
        raise ValueError('epsilon must be non-negative.')
    operand = sanitize_input(operand, get_data_type(operand))  
    return mean_variance_normalization(operand, epsilon, use_stats_across_channels, do_variance_scaling, name)

@typemap
def custom_proxy_op(custom_op, output_shape, output_data_type, *operands, **kw_name):
    '''
    A proxy node that helps saving a model with different number of operands. 

    Example:

    Args:

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import custom_proxy_op
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]
    if isinstance(operands, tuple):
        operands = list(operands)
    operands_unfold = []
    for o in operands:
        if hasattr(o, 'outputs') and len(o.outputs) > 1:
            operands_unfold += o.outputs
        else:
            operands_unfold += [o]

    name = (lambda name='': (name))(**kw_name) # Python 2.7 does not allow (*inputs, name='')
    output_dtype = sanitize_dtype_cntk(output_data_type)
    return custom_proxy_op(operands_unfold, custom_op, output_shape, output_dtype, name)