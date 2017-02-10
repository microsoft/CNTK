# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from __future__ import print_function
import numpy as np
import numbers
from numbers import Number
from . import sequence
from .functions import CloneMethod, Function, load_model
from .variables import Variable, Parameter, Constant
from ..utils import sanitize_input, sanitize_shape, get_data_type, sanitize_axis, sanitize_dynamic_axes, typemap
from ..axis import Axis

@typemap
def combine(operands, name=''):
    '''
     Create a new Function instance which just combines the outputs of the specified list of
     'operands' Functions such that the 'Outputs' of the new 'Function' are union of the
     'Outputs' of each of the specified 'operands' Functions. E.g. When creating a classification
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

    Args:
        operands (list): list of functions or their variables to combine
        name (str, optional): the name of the Combine Function in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import combine
    return combine(operands, name)

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
     primtive. Users still have the ability to peek at the underlying Function graph that implements
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

##########################################################################
# loss and evaluation ops
##########################################################################

@typemap
def cosine_distance(x, y, name=''):
    '''
    Computes the cosine distance between ``x`` and ``y``:

    Example:
        >>> a = np.asarray([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]).reshape(3,2,2)
        >>> b = np.asarray([1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1]).reshape(3,2,2)
        >>> x = C.input_variable(shape=(2,))
        >>> y = C.input_variable(shape=(2,))
        >>> np.round(C.cosine_distance(x,y).eval({x:a,y:b}),5)
        array([[-1.,  1.],
               [ 1.,  0.],
               [ 0., -1.]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import cosine_distance
    dtype = get_data_type(x, y)
    x = sanitize_input(x, dtype)
    y = sanitize_input(y, dtype)
    return cosine_distance(x, y, name)

@typemap
def binary_cross_entropy(output, target, name=''):
    r'''
    Computes the binary cross entropy (aka logistic loss) between the ``output`` and ``target``.

    Example:
        TBA

    Args:
        output: the computed posterior probability for a variable to be 1 from the network (typ. a ``sigmoid``)
        target: ground-truth label, 0 or 1
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import binary_cross_entropy
    dtype = get_data_type(output, target)
    output = sanitize_input(output, dtype)
    target = sanitize_input(target, dtype)
    return binary_cross_entropy(output, target, name)

@typemap
def weighted_binary_cross_entropy(output, target, weight, name=''):
    r'''
    This operation computes the weighted binary cross entropy (aka logistic loss) between the ``output`` and ``target``.

    Example:
        TBA

    Args:
        output: the computed posterior probability from the network
        target: ground-truth label, 0 or 1
        weight: weight of each example
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import weighted_binary_cross_entropy
    dtype = get_data_type(output, target, weight)
    output = sanitize_input(output, dtype)
    target = sanitize_input(target, dtype)
    weight = sanitize_input(weight, dtype)
    return weighted_binary_cross_entropy(output, target, weight, name)

@typemap
def cross_entropy_with_softmax(output_vector, target_vector, axis=-1, name=''):
    r'''
    This operation computes the cross entropy between the ``target_vector`` and
    the softmax of the ``output_vector``. The elements of ``target_vector``
    have to be non-negative and should sum to 1. The ``output_vector`` can
    contain any values. The function will internally compute the softmax of
    the ``output_vector``. Concretely,

    :math:`\mathrm{softmax}(x)=\left[\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\ldots\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\right]`

    :math:`\mathrm{cross\_entropy\_with\_softmax}(o, t) = -\sum_{i} t_i \log(\mathrm{softmax}(o)_i)`

    with the understanding that the implementation can use equivalent formulas
    for efficiency and numerical stability.

    Example:
        >>> C.cross_entropy_with_softmax([[1., 1., 1., 50.]], [[0., 0., 0., 1.]]).eval()
        array([[ 0.]], dtype=float32)

        >>> C.cross_entropy_with_softmax([[1., 2., 3., 4.]], [[0.35, 0.15, 0.05, 0.45]]).eval()
        array([[ 1.84019]], dtype=float32)

    Args:
        output_vector: the unscaled computed output values from the network
        target_vector: usually it is one-hot vector where the hot bit
         corresponds to the label index. But it can be any probability
         distribution over the labels.
        axis (int or :class:`~cntk.axis.Axis`): axis along which the cross
         entropy will be computed.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import cross_entropy_with_softmax
    dtype = get_data_type(output_vector, target_vector)
    output_vector = sanitize_input(output_vector, dtype)
    target_vector = sanitize_input(target_vector, dtype)
    axis = sanitize_axis(axis)
    return cross_entropy_with_softmax(output_vector, target_vector, axis, name)


@typemap
def squared_error(output, target, name=''):
    '''
    This operation computes the sum of the squared difference between elements
    in the two input matrices. The result is a scalar (i.e., one by one matrix).
    This is often used as a training criterion.

    Example:
        >>> i1 = C.input_variable((1,2))
        >>> i2 = C.input_variable((1,2))
        >>> C.squared_error(i1,i2).eval({i1:np.asarray([[[[2., 1.]]]], dtype=np.float32), i2:np.asarray([[[[4., 6.]]]], dtype=np.float32)})
        array([[ 29.]], dtype=float32)

        >>> C.squared_error(i1,i2).eval({i1:np.asarray([[[[1., 2.]]]], dtype=np.float32), i2:np.asarray([[[[1., 2.]]]], dtype=np.float32)})
        array([[ 0.]], dtype=float32)

    Args:
        output: the output values from the network
        target: it is usually a one-hot vector where the hot bit
         corresponds to the label index
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import squared_error
    dtype = get_data_type(output, target)
    output = sanitize_input(output, dtype)
    target = sanitize_input(target, dtype)
    return squared_error(output, target, name)

@typemap
def lambda_rank(output, gain, group, name=''):
    r'''
    Groups samples according to ``group``, sorts
    them within each group based on ``output`` and
    computes the Normalized Discounted Cumulative Gain
    (NDCG) at infinity for each group. Concretely,
    the Discounted Cumulative Gain (DCG) at infinity is:

    :math:`\mathrm{DCG_{\infty}}()=\sum_{i=0}^{\infty} \frac{gain_{(i)}}{\log(i+2)}`

    where :math:`gain_{(i)}` means the gain of the :math:`i`-th ranked sample.

    The NDCG is just the DCG  divided by the maximum achievable DCG (obtained
    by placing the samples with the largest gain at the top of the ranking).

    Samples in the same group must appear in order of decreasing gain.

    It returns 1 minus the average NDCG across all the groups in the minibatch
    multiplied by 100 times the number of samples in the minibatch.

    In the backward direction it back-propagates LambdaRank gradients.

    Example:
        >>> group = C.input_variable((1,))
        >>> score = C.input_variable((1,), needs_gradient=True)
        >>> gain  = C.input_variable((1,))
        >>> g = np.array([1, 1, 2, 2], dtype=np.float32).reshape(4,1,1)
        >>> s = np.array([1, 2, 3, 4], dtype=np.float32).reshape(4,1,1)
        >>> n = np.array([7, 1, 3, 1], dtype=np.float32).reshape(4,1,1)
        >>> f = C.lambda_rank(score, gain, group)
        >>> np.round(f.grad({score:s, gain:n, group: g}, wrt=[score])[0],4)
        array([[[-0.2121]],
        <BLANKLINE>
               [[ 0.2121]],
        <BLANKLINE>
               [[-0.1486]],
        <BLANKLINE>
               [[ 0.1486]]], dtype=float32)

    Args:
        output: score of each sample
        gain: gain of each sample
        group: group of each sample
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import lambda_rank
    dtype = get_data_type(output, gain, group)
    output = sanitize_input(output, dtype)
    gain = sanitize_input(gain, dtype)
    group = sanitize_input(group, dtype)
    return lambda_rank(output, gain, group, name)


@typemap
def ndcg_at_1(output, gain, group, name=''):
    r'''
    Groups samples according to ``group``, sorts
    them within each group based on ``output`` and
    computes the Normalized Discounted Cumulative Gain
    (NDCG) at 1 for each group. Concretely,
    the NDCG at 1 is:

    :math:`\mathrm{NDCG_1} = \frac{gain_{(1)}}{\max_i gain_i}`

    where :math:`gain_{(1)}` means the gain of the first ranked sample.

    Samples in the same group must appear in order of decreasing gain.

    It returns the average NDCG at 1 across all the groups in the minibatch
    multiplied by 100 times the number of samples in the minibatch.

    This is a forward-only operation, there is no gradient for it.

    Example:
        >>> group = C.input_variable((1,))
        >>> score = C.input_variable((1,))
        >>> gain  = C.input_variable((1,))
        >>> g = np.array([1, 1, 2, 2], dtype=np.float32).reshape(4,1,1)
        >>> s = np.array([2, 1, 3, 1], dtype=np.float32).reshape(4,1,1)
        >>> n = np.array([7, 1, 3, 1], dtype=np.float32).reshape(4,1,1)
        >>> C.ndcg_at_1(score, gain, group).eval({score:s, gain:n, group: g})
        array(400.0, dtype=float32)

    Args:
        output: score of each sample
        gain: gain of each sample
        group: group of each sample
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import ndcg_at_1
    dtype = get_data_type(output, gain, group)
    output = sanitize_input(output, dtype)
    gain = sanitize_input(gain, dtype)
    group = sanitize_input(group, dtype)
    return ndcg_at_1(output, gain, group, name)


@typemap
def classification_error(output_vector, target_vector, axis=-1, topN=1, name=''):
    '''
    This operation computes the classification error. It finds the index of the highest
    value in the output_vector and compares it to the actual ground truth label
    (the index of the hot bit in the target vector). The result is a scalar
    (i.e., one by one matrix). This is often used as an evaluation criterion.
    It cannot be used as a training criterion though since the gradient is not
    defined for it.

    Example:
        >>> C.classification_error([[1., 2., 3., 4.]], [[0., 0., 0., 1.]]).eval()
        array([[ 0.]], dtype=float32)

        >>> C.classification_error([[1., 2., 3., 4.]], [[0., 0., 1., 0.]]).eval()
        array([[ 1.]], dtype=float32)

        >>> # Note that non-1 values are treated as 0
        >>> C.classification_error([[1., 2., 3., 4.]], [[5., 0., 1., 0.]]).eval()
        array([[ 1.]], dtype=float32)

    Args:
        output_vector: the output values from the network
        target_vector: it is one-hot vector where the hot bit corresponds to
         the label index.
        axis (int or :class:`~cntk.axis.Axis`): axis along which the
         classification error will be computed.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import classification_error
    dtype = get_data_type(output_vector, target_vector)
    output_vector = sanitize_input(output_vector, dtype)
    target_vector = sanitize_input(target_vector, dtype)
    axis = sanitize_axis(axis)
    return classification_error(output_vector, target_vector, topN, axis, name)

@typemap
def edit_distance_error(input_a, input_b, subPen=0, delPen=0, insPen=0, squashInputs=False, samplesToIgnore=[], name=''):
    '''
    Edit distance error evaluation node with the option of specifying penalty of substitution, deletion and insertion, as well as squashing the input sequences and ignoring certain samples.
    Using the classic DP algorithm as described in https://en.wikipedia.org/wiki/Edit_distance, adjusted to take into account the penalties.

    Each sequence in the inputs is expected to be a matrix. Prior to computation of the edit distance, the operation extracts the indices of maximum element in each column.
    For example, a sequence matrix
    1 2 9 1
    3 0 3 2
    will be represented as the vector of labels (indices) as [1, 0, 0, 1], on which edit distance will be actually evaluated.

    The node allows to squash sequences of repeating labels and ignore certain labels. For example, if squashInputs is true and samplesToIgnore contains label '-' then
    given first input sequence as s1="1-12-" and second as s2="-11--122" the edit distance will be computed against s1' = "112" and s2' = "112".

    The returned error is computed as: EditDistance(s1,s2) * length(s1') / length(s1)

    Just like ClassificationError and other evaluation nodes, when used as an evaluation criterion, the SGD process will aggregate all values over an epoch and report the average, i.e. the error rate.
    Primary objective of this node is for error evaluation of CTC training, see formula (1) in "Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks", http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf

    Example:
        i1 = cntk.input_variable(shape=(2,))
        i2 = cntk.input_variable(shape=(2,))
        arguments = {i1 : [[1, 3], [2, 0]], i2 : [[2, 0], [2, 0]]}
        a = edit_distance_error(i1, i2, 0, 1, 1, True, [1])
        print(a.eval(arguments))

    Args:
        input_a: first input sequence
        input_b: second input sequence
        subPen, delPen, insPen: substitution, deletion and insertion penalties
        squashInputs: whether to merge sequences of identical samples (in both input sequences). If true and samplesToIgnore contains label '-' then
                given first input sequence as s1="a-ab-" and second as s2="-aa--abb" the edit distance will be computed against s1' = "aab" and s2' = "aab".
        samplesToIgnore: list of samples to ignore during edit distance evaluation (in both sequences)
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import edit_distance_error
    dtype = get_data_type(input_a, input_b)
    input_a = sanitize_input(input_a, dtype)
    input_b = sanitize_input(input_b, dtype)
    return edit_distance_error(input_a, input_b, subPen, delPen, insPen, squashInputs, samplesToIgnore, name)


##########################################################################
# convolution ops
##########################################################################

@typemap
def convolution(convolution_map, operand, strides=(1,), sharing=[True],
                auto_padding=[True], lower_pad=(0,), upper_pad=(0,), transpose=False,
                max_temp_mem_size_in_samples=0, name=''):
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
    The first dimension, :math:`O`, is the nunber of convolution filters (i.e. the number of
    channels in the output). The second dimension, :math:`I`, must match the number of channels in the input.
    The last n dimensions are the spatial extent of the filter. I.e. for each output position, a vector of
    dimension :math:`O` is computed. Hence, the total number of filter parameters is :math:`O \\times I \\times m_1 \\times m_2 \\times \\ldots \\times m_n`


    Example:
        >>> img = np.reshape(np.arange(25.0, dtype = np.float32), (1, 5, 5))
        >>> x = C.input_variable(img.shape)
        >>> filter = np.reshape(np.array([2, -1, -1, 2], dtype = np.float32), (1, 2, 2))
        >>> kernel = C.constant(value = filter)
        >>> np.round(C.convolution(kernel, x, auto_padding = [False]).eval({x: [img]}),5)
        array([[[[[  6.,   8.,  10.,  12.],
                  [ 16.,  18.,  20.,  22.],
                  [ 26.,  28.,  30.,  32.],
                  [ 36.,  38.,  40.,  42.]]]]], dtype=float32)

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
        lower_pad: precise lower padding for each input dimension.
        upper_pad : precise upper padding for each input dimension.
        transpose (bool): set to true for deconvolution.
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
    strides = sanitize_shape(strides)
    lower_pad = sanitize_shape(lower_pad)
    upper_pad = sanitize_shape(upper_pad)
    return convolution(convolution_map, operand, strides, sharing, auto_padding,
                       lower_pad, upper_pad, transpose,
                       max_temp_mem_size_in_samples, name)


@typemap
def roipooling(conv_feature_map, rois, roi_output_shape, name=''):
    '''
    The ROI (Region of Interest) pooling operation pools over sub-regions of an input volume and produces
    a fixed sized output volume regardless of the ROI size. It is used for example for object detection.

    Each input image has a fixed number of regions of interest, which are specified as bounding boxes (x, y, w, h)
    that are relative to the image size [W x H]. This operation can be used as a replacement for the final
    pooling layer of an image classification network (as presented in Fast R-CNN and others).

    Args:
        conv_feature_map: a convolutional feature map as the input volume ([W x H x C x N]).
        rois: the coordinates of the ROIs per image ([4 x roisPerImage x N]), each ROI is (x, y, w, h) relative to original image size.
        roi_output_shape: dimensions (width x height) of the ROI pooling output shape
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import roipooling
    conv_feature_map = sanitize_input(conv_feature_map)
    rois = sanitize_input(rois)
    roi_output_shape = sanitize_shape(roi_output_shape)
    return roipooling(conv_feature_map, rois, roi_output_shape, name)


from cntk.cntk_py import PoolingType_Max, PoolingType_Average
MAX_POOLING = PoolingType_Max
AVG_POOLING = PoolingType_Average

@typemap
def pooling(operand, pooling_type, pooling_window_shape, strides=(1,), auto_padding=[False],
            lower_pad=(0,), upper_pad=(0,), name=''):
    '''
    The pooling operations compute a new tensor by selecting the maximum or average value in the pooling input.
    In the case of average pooling with padding, the average is only over the valid region.

    N-dimensional pooling allows to create max or average pooling of any dimensions, stride or padding.

    Example:
        >>> img = np.reshape(np.arange(16, dtype = np.float32), [1, 4, 4])
        >>> x = C.input_variable(img.shape)
        >>> C.pooling(x, C.AVG_POOLING, (2,2), (2,2)).eval({x : [img]})
        array([[[[[  2.5,   4.5],
                  [ 10.5,  12.5]]]]], dtype=float32)
        >>> C.pooling(x, C.MAX_POOLING, (2,2), (2,2)).eval({x : [img]})
        array([[[[[  5.,   7.],
                  [ 13.,  15.]]]]], dtype=float32)

    Args:
        operand: pooling input
        pooling_type: one of :const:`~cntk.ops.MAX_POOLING` or :const:`~cntk.ops.AVG_POOLING`
        pooling_window_shape: dimensions of the pooling window
        strides (default 1): strides.
        auto_padding: automatic padding flags for each input dimension.
        lower_pad: precise lower padding for each input dimension
        upper_pad: precise upper padding for each input dimension
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import pooling
    operand = sanitize_input(operand)
    pooling_window_shape = sanitize_shape(pooling_window_shape)
    strides = sanitize_shape(strides)
    lower_pad = sanitize_shape(lower_pad)
    upper_pad = sanitize_shape(upper_pad)
    return pooling(operand, pooling_type, pooling_window_shape, strides, auto_padding,
                   lower_pad, upper_pad, name)


MAX_UNPOOLING = PoolingType_Max
@typemap
def unpooling(operand, pooling_input, unpooling_type, unpooling_window_shape, strides=(1,), auto_padding=[False],
            lower_pad=(0,), upper_pad=(0,), name=''):
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
	array([[[[[  0.,   0.,   0.,   0.],
		  [  0.,   5.,   0.,   7.],
		  [  0.,   0.,   0.,   0.],
		  [  0.,  13.,   0.,  15.]]]]], dtype=float32)

    Args:
        operand: unpooling input
        pooling_input: input to the corresponding pooling operation
        unpooling_type: only :const:`~cntk.ops.MAX_UNPOOLING` is supported now
        unpooling_window_shape: dimensions of the unpooling window
        strides (default 1): strides.
        auto_padding: automatic padding flags for each input dimension.
        lower_pad: precise lower padding for each input dimension
        upper_pad: precise upper padding for each input dimension
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import unpooling
    operand = sanitize_input(operand)
    pooling_input = sanitize_input(pooling_input)
    unpooling_window_shape = sanitize_shape(unpooling_window_shape)
    strides = sanitize_shape(strides)
    lower_pad = sanitize_shape(lower_pad)
    upper_pad = sanitize_shape(upper_pad)
    return unpooling(operand, pooling_input, unpooling_type,
                     unpooling_window_shape, strides, auto_padding,
                     lower_pad, upper_pad, name)


@typemap
def batch_normalization(operand, scale, bias, running_mean, running_inv_std, spatial,
                        normalization_time_constant=5000, blend_time_constant=0,
                        epsilon=0.00001, use_cudnn_engine=False, name='', running_count=None):
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
        use_cudnn_engine(bool, default True):
        name (str, optional): the name of the Function instance in the network
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
                               epsilon, use_cudnn_engine, name)

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

@typemap
def plus(left, right, *more, **kwname):
    '''
    The output of this operation is the sum of the two or more input tensors. It supports broadcasting.

    Example:
        >>> C.plus([1, 2, 3], [4, 5, 6]).eval()
        array([ 5.,  7.,  9.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10]).eval()
        array([ 5.,  6.,  7.,  8.,  9.], dtype=float32)

        >>> C.plus([-5, -4, -3, -2, -1], [10], [3, 2, 3, 2, 3]).eval()
        array([  8.,   8.,  10.,  10.,  12.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        *more: additional summands
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': name)(**kwname) # Python 2.7 does not allow (left, right, *more, name='')
    if more: # if additional operands then recurse
        return plus(plus(left, right, *more[:-1], name=''), more[-1], name=name)
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
def element_times(left, right, *more, **kwname):
    '''
    The output of this operation is the element-wise product of the two or more input
    tensors. It supports broadcasting.

    Example:
        >>> C.element_times([1., 1., 1., 1.], [0.5, 0.25, 0.125, 0.]).eval()
        array([ 0.5  ,  0.25 ,  0.125,  0.   ], dtype=float32)

        >>> C.element_times([5., 10., 15., 30.], [2.]).eval()
        array([ 10.,  20.,  30.,  60.], dtype=float32)

    Args:
        left: left side tensor
        right: right side tensor
        *more: additional factors
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': name)(**kwname) # Python 2.7 does not allow (left, right, *more, name='')
    if more: # if additional operands then recurse
        return element_times(element_times(left, right, *more[:-1], name=''), more[-1], name=name)
    from cntk.cntk_py import element_times as cntk_py_element_times
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_element_times(left, right, name)


@typemap
def element_max(left, right, *more, **kwname):
    '''
    The output of this operation is the element-wise max of the two or more input
    tensors. It supports broadcasting.

    Args:
        left: left side tensor
        right: right side tensor
        *more: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': name)(**kwname) # Python 2.7 does not allow (left, right, *more, name='')
    if more: # if additional operands then recurse
        return element_max(element_max(left, right, *more[:-1], name=''), more[-1], name=name)
    gt = greater(left, right)
    # TODO: use as_block()
    return element_select(gt, left, right, name)


@typemap
def element_min(left, right, *more, **kwname):
    '''
    The output of this operation is the element-wise min of the two or more input
    tensors. It supports broadcasting.

    Args:
        left: left side tensor
        right: right side tensor
        *more: additional inputs
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': name)(**kwname) # Python 2.7 does not allow (left, right, *more, name='')
    if more: # if additional operands then recurse
        return element_min(element_min(left, right, *more[:-1], name=''), more[-1], name=name)
    lt = less(left, right)
    # TODO: use as_block()
    return element_select(lt, left, right, name)


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


@typemap
def log_add_exp(left, right, *more, **kwname):
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
        left: left side tensor
        right: right side tensor
        *more: additional summands
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    name = (lambda name='': name)(**kwname) # Python 2.7 does not allow (left, right, *more, name='')
    if more: # if additional operands then recurse
        return log_add_exp(log_add_exp(left, right, *more[:-1], name=''), more[-1], name=name)
    from cntk.cntk_py import log_add_exp as cntk_py_log_add_exp
    dtype = get_data_type(left, right)
    left = sanitize_input(left, dtype)
    right = sanitize_input(right, dtype)
    return cntk_py_log_add_exp(left, right, name)


@typemap
def times(left, right, output_rank=1, infer_input_rank_to_map=-1, name=''):
    '''
    The output of this operation is the matrix product of the two input matrices.
    It supports broadcasting. Sparse is supported in the right operand, if it is a matrix.
    The operator '@' has been overloaded such that in Python 3.5 and later X @ W equals times(X, W).

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
        output_rank (int): in case we have tensors as arguemnts, output_rank represents
            the number of axes to be collapsed in order to transform the tensors
            into matrices, perform the operation and then reshape back (explode the axes)
        infer_input_rank_to_map ('int'): meant for internal use only. Always use default value
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
def sigmoid(x, name=''):
    '''
    Computes the element-wise sigmoid of ``x``:

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

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
def softmax(x, axis=None, name=''):
    r'''
    Computes the gradient of :math:`f(z)=\log\sum_i\exp(z_i)` at z=``x``. Concretely,

    :math:`\mathrm{softmax}(x)=\left[\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\ldots\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\right]`

    with the understanding that the implementation can use equivalent formulas
    for efficiency and numerical stability.

    The output is a vector of non-negative numbers that sum to 1 and can
    therefore be interpreted as probabilities for mutually exclusive outcomes
    as in the case of multiclass classification.

    If ``axis`` is given, the softmax will be computed along that axis.

    Example:
        >>> C.softmax([[1, 1, 2, 3]]).eval()
        array([[ 0.082595,  0.082595,  0.224515,  0.610296]], dtype=float32)

        >>> C.softmax([1, 1]).eval()
        array([ 0.5,  0.5], dtype=float32)

        >>> C.softmax([[[1, 1], [3, 5]]], axis=-1).eval()
        array([[[ 0.5     ,  0.5     ],
                [ 0.119203,  0.880797]]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the softmax operation will be performed
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import softmax
    x = sanitize_input(x)
    # softmax over a specific axis: implemented explicitly
    if axis is not None:
        from cntk.cntk_py import reduce_log_sum, exp, minus
        axis = sanitize_axis(axis)
        Z = reduce_log_sum(x, axis)  # log denominator
        # TODO: use as_block()
        return exp(x - Z.output(), name) # this is the softmax
        # (note: we need .output() here since the automatisms available outside are not available in here)
    # softmax over all elements
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
def exp(x, name=''):
    '''
    Computes the element-wise exponential of ``x``:

    :math:`\exp(x) = {e^x}`

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

    :math:`sqrt(x) = {\sqrt[2]{x}}`

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
        retrun NaN
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


##########################################################################
# recurrent ops
# TODO: do these belong into .sequence?
##########################################################################

@typemap
def future_value(x, initial_state=None, time_step=1, name=''):
    '''
    This function returns the future value w.r.t. ``x``. It is most often used when
    creating RNNs. The resulting tensor has the same shape as the input but is
    the next logical sample. The ``time_step`` parameter is the number of steps
    to look into the future and is 1 by default. If there is no future value (i.e.
    the current sample is the last one in the tensor) then the ``initial_state``
    value is returned.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> # Create one sequence with 4 tensors of shape (3, 2)
        >>> x0 = np.reshape(np.arange(24,dtype=np.float32),(1,4,3,2))
        >>> y = C.future_value(x) # using initial state of 0 by default
        >>> y.eval({x:x0})
        array([[[[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]],
        <BLANKLINE>
                [[ 12.,  13.],
                 [ 14.,  15.],
                 [ 16.,  17.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]],
        <BLANKLINE>
                [[  0.,   0.],
                 [  0.,   0.],
                 [  0.,   0.]]]], dtype=float32)

    Args:
        x: the tensor (or its name) from which the future value is obtained.
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the future (default 1)
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from ..utils import sanitize_dtype_cntk
    from ..cntk_py import Constant
    from cntk.cntk_py import future_value

    if initial_state is None:
        initial_state = Constant.scalar(sanitize_dtype_cntk(np.float32), 0.0)

    x = sanitize_input(x)
    return future_value(x, initial_state, time_step, name)


@typemap
def past_value(x, initial_state=None, time_step=1, name=''):
    '''
    This function returns the past value w.r.t. ``x``. It is most often used when
    creating RNNs. The resulting tensor has the same shape as the input but is
    the previous logical sample. The ``time_step`` parameter is the number of steps
    to look into the past and is 1 by default. If there is no past value (i.e.
    the current sample is the first one in the tensor)  then the ``initial_state``
    value is returned.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> # Create one sequence with 4 tensors of shape (3, 2)
        >>> x0 = np.reshape(np.arange(24,dtype=np.float32),(1,4,3,2))
        >>> y = C.past_value(x) # using initial state of 0 by default
        >>> y.eval({x:x0})
        array([[[[  0.,   0.],
                 [  0.,   0.],
                 [  0.,   0.]],
        <BLANKLINE>
                [[  0.,   1.],
                 [  2.,   3.],
                 [  4.,   5.]],
        <BLANKLINE>
                [[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]],
        <BLANKLINE>
                [[ 12.,  13.],
                 [ 14.,  15.],
                 [ 16.,  17.]]]], dtype=float32)

    Args:
        x: the tensor (or its name) from which the past value is obtained
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the past (default 1)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from ..utils import sanitize_dtype_cntk
    from ..cntk_py import Constant
    from cntk.cntk_py import past_value

    if initial_state is None:
        initial_state = Constant.scalar(sanitize_dtype_cntk(np.float32), 0.0)

    x = sanitize_input(x)
    return past_value(x, initial_state, time_step, name)


# TODO: does this belong into .sequence?
@typemap
def optimized_rnnstack(operand, weights, hidden_size, num_layers,
                       bidirectional=False, recurrent_op='lstm', name=''):
    '''
    An RNN implementation that uses the primitives in cuDNN.
    If cuDNN is not available it fails.

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
        >>> from _cntk_py import InferredDimension, constant_initializer
        >>> W = C.parameter((InferredDimension,4), constant_initializer(0.1))
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
    if recurrent_op not in set(['lstm','gru','relu','tanh']):
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
        array([[[[ 0.,  1.,  2.],
                 [ 3.,  4.,  5.]]]], dtype=float32)

    Args:
        x: tensor to be reshaped
        shape (tuple): a tuple defining the resulting shape
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
        end_axis = Axis.end_static_axis()

    # Pass begin_axis as the end_axis and vice versa to account for
    # the automatic shape reversal across the python SWIG boundary
    def sanitize_reshape_axis(axis):
        if isinstance(axis, numbers.Integral):
            axis = Axis(axis)

        if not axis.is_static_axis:
            return axis

        if (axis ==  Axis.end_static_axis()):
            return Axis(0)
        elif (axis == Axis(0)):
            return Axis.end_static_axis()
        else:
            return Axis(-axis.static_axis_index())

    internal_reshape_begin_axis = sanitize_reshape_axis(end_axis)
    internal_reshape_end_axis = sanitize_reshape_axis(begin_axis)

    return reshape(x, shape, internal_reshape_begin_axis, internal_reshape_end_axis, name)


@typemap
def transpose(x, axis1=0, axis2=1, name=''):
    '''
    Swaps two axes of the tensor. The output tensor has the same data but with
    ``axis1`` and ``axis2`` swapped.

    Example:
        >>> C.transpose([[[0,1],[2,3],[4,5]]], 1, 2).eval()
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
def slice(x, axis, begin_index, end_index, name=''):
    '''
    Slice the input along an axis.

    Example:
        >>> # Slice using input variable
        >>> # create 2x3 matrix
        >>> x1 = C.input_variable((2,3))
        >>> # slice index 1 (second) at first axis
        >>> C.slice(x1, 0, 1, 2).eval({x1: np.asarray([[[[1,2,-3],
        ...                                              [4, 5, 6]]]],dtype=np.float32)})
        array([[[[ 4.,  5.,  6.]]]], dtype=float32)
        >>> # slice index 0 (first) at second axis
        >>> C.slice(x1, 1, 0, 1).eval({x1: np.asarray([[[[1,2,-3],
        ...                                              [4, 5, 6]]]],dtype=np.float32)})
        array([[[[ 1.],
                 [ 4.]]]], dtype=float32)

        >>> #slice using constant
        >>> data = np.asarray([[1, 2, -3],
        ...                     [4, 5,  6]], dtype=np.float32)
        >>> x = C.constant(value=data)
        >>> C.slice(x, 0, 1, 2).eval()
        array([[ 4.,  5.,  6.]], dtype=float32)
        >>> C.slice(x, 1, 0, 1).eval()
        array([[ 1.],
               [ 4.]], dtype=float32)

    NumPy's way of slicing works, too:

    Example:
        #TODO: Make following lines work. Uncomment when done
        #>>> x1[1].eval()
        #array([[ 4.,  5.,  6.]], dtype=float32)
        #>>> x1[:,:2,:].eval()
        #array([[ 1.,  2.],
        #         [ 4.,  5.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which ``begin_index`` and ``end_index``
         will be used. If it is of type int it will be used as a static axis.
        begin_index (int): the index along axis where the slicing starts
        end_index (int): the index along axis where the slicing ends
        name (str, optional): the name of the Function instance in the network

    See also:
        Indexing in NumPy: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import slice
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return slice(x, axis, begin_index, end_index, name)

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

##########################################################################
# reduction ops
##########################################################################


@typemap
def reduce_sum(x, axis=None, name=''):
    '''
    Computes the sum of the input tensor's elements across one axis. If the axis parameter
    is not specified then the sum will be computed over all static axes, which is 
    equivalent with specifying ``axis=Axis.all_static_axes()``. If 
    ``axis=Axis.all_axes()``, the output is a scalar which is the sum of all the 
    elements in the minibatch.

    Example:
        >>> x = C.input_variable((2,2))
        >>> # create a batch of 2 sequences each containing 2 2x2 matrices 
        >>> x0 = np.arange(16,dtype=np.float32).reshape(2,2,2,2)
        >>> # reduce over all static axes
        >>> C.reduce_mean(x).eval({x:x0})
        array([[  1.5,   5.5],
               [  9.5,  13.5]], dtype=float32)
        >>> # reduce over specified axes
        >>> C.reduce_mean(x,axis=0).eval({x:x0})
        array([[[[  1.,   2.]],
        <BLANKLINE>
                [[  5.,   6.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[  9.,  10.]],
        <BLANKLINE>
                [[ 13.,  14.]]]], dtype=float32)
        >>> C.reduce_mean(x,axis=1).eval({x:x0})
        array([[[[  0.5],
                 [  2.5]],
        <BLANKLINE>
                [[  4.5],
                 [  6.5]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[  8.5],
                 [ 10.5]],
        <BLANKLINE>
                [[ 12.5],
                 [ 14.5]]]], dtype=float32)
        >>> # reduce over all axes
        >>> np.round(C.reduce_mean(x, axis=C.Axis.all_axes()).eval({x:x0}),5)
        7.5
        >>> # reduce over all axes when the batch has sequences of different length
        >>> x1 = np.arange(4,dtype=np.float32).reshape(1,2,2)
        >>> x2 = np.arange(12,dtype=np.float32).reshape(3,2,2)
        >>> np.round(C.reduce_mean(x, axis=C.Axis.all_axes()).eval({x:[x1,x2]}),5)
        4.5
        >>> (np.sum(x1)+np.sum(x2))/(x1.size+x2.size)
        4.5

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_sum
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_sum(x, axis, name)


@typemap
def reduce_log_sum_exp(x, axis=None, name=''):
    '''
    Computes the log of the sum of the exponentiations of the input tensor's
    elements across the specified axis.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> val = np.reshape(np.arange(6.0, dtype=np.float32), (3,2))
        >>> lse = C.reduce_log_sum_exp(x)
        >>> lse.eval({x:[val]})
        array([[ 5.456193]], dtype=float32)
        >>> np.log(np.sum(np.exp(val)))
        5.4561934

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str): the name of the Function instance in the network

    See also:
        :func:`~cntk.ops.reduce_sum` for more details and examples.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    # TODO: rename V2 API function as well from reduce_log_sum() to reduce_log_sum_exp()
    from cntk.cntk_py import reduce_log_sum
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_log_sum(x, axis, name)


@typemap
def reduce_mean(x, axis=None, name=''):
    '''
    Computes the mean of the input tensor's elements across the specified axis.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[5, 20],[30, 40],[55, 60]]

        >>> C.reduce_mean(data, 0).eval()
        array([[ 30.,  40.]], dtype=float32)

        >>> C.reduce_mean(data, 1).eval()
        array([[ 12.5],
               [ 35. ],
               [ 57.5]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str, optional): the name of the Function instance in the network

    See also:
        :func:`~cntk.ops.reduce_sum` for more details and examples.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_mean
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_mean(x, axis, name)


@typemap
def reduce_max(x, axis=None, name=''):
    '''
    Computes the max of the input tensor's elements across the specified axis.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 20],[30, 40],[50, 60]]

        >>> C.reduce_max(data, 0).eval()
        array([[ 50.,  60.]], dtype=float32)

        >>> C.reduce_max(data, 1).eval()
        array([[ 20.],
               [ 40.],
               [ 60.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str): the name of the Function instance in the network

    See also:
        :func:`~cntk.ops.reduce_sum` for more details and examples.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_max
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_max(x, axis, name)


@typemap
def reduce_min(x, axis=None, name=''):
    '''
    Computes the min of the input tensor's elements across the specified axis.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[10, 20],[30, 40],[50, 60]]

        >>> C.reduce_min(data, 0).eval()
        array([[ 10.,  20.]], dtype=float32)

        >>> C.reduce_min(data, 1).eval()
        array([[ 10.],
               [ 30.],
               [ 50.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str): the name of the Function instance in the network

    See also:
        :func:`~cntk.ops.reduce_sum` for more details and examples.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_min
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_min(x, axis, name)


@typemap
def reduce_prod(x, axis=None, name=''):
    '''
    Computes the min of the input tensor's elements across the specified axis.

    Example:
        >>> # create 3x2 matrix in a sequence of length 1 in a batch of one sample
        >>> data = [[1, 2],[3, 4],[5, 6]]

        >>> C.reduce_prod(data, 0).eval()
        array([[ 15.,  48.]], dtype=float32)

        >>> C.reduce_prod(data, 1).eval()
        array([[  2.],
               [ 12.],
               [ 30.]], dtype=float32)

    Args:
        x: input tensor
        axis (int or :class:`~cntk.axis.Axis`): axis along which the reduction will be performed
        name (str): the name of the Function instance in the network

    See also:
        :func:`~cntk.ops.reduce_sum` for more details and examples.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import reduce_prod
    x = sanitize_input(x)
    axis = sanitize_axis(axis)
    return reduce_prod(x, axis, name)


#######################################################################
# training ops
#######################################################################

@typemap
def random_sample(weights, num_samples, allow_duplicates, name=''):
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

    Returns:
        :class:`~cntk.ops.functions.Function`

    '''
    from cntk.cntk_py import random_sample
    weights = sanitize_input(weights)

    return random_sample(weights, num_samples, allow_duplicates, name)


@typemap
def random_sample_inclusion_frequency(
    weights, 
    num_samples, 
    allow_duplicates, 
    name=''):
    '''
    For weighted sampling with the specifed sample size (`num_samples`)
    this operation computes the expected number of occurences of each class
    in the the sampled set. In case of sampling without replacement
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
        name)


@typemap
def dropout(x, dropout_rate=0.0, name=''):
    '''
    Each element of the input is independently set to 0 with probabily ``dropout_rate``
    or to 1 / (1 - ``dropout_rate``) times its original value (with probability 1-``dropout_rate``).
    Dropout is a good way to reduce overfitting.

    This behavior only happens during training. During inference dropout is a no-op.
    In the paper that introduced dropout it was suggested to scale the weights during inference
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
        name (:class:str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        raise ValueError('dropout_rate must be in the interval [0,1)')

    from cntk.cntk_py import dropout
    x = sanitize_input(x)

    return dropout(x, dropout_rate, name)

##########################################################################
# variables_and_parameters ops
##########################################################################

from cntk.device import use_default_device
from cntk.axis import Axis

# TODO: if we end up using only factory methods, we should get rid of the
# class Variable in variables.py


@typemap
def input_variable(shape, dtype=np.float32, needs_gradient=False, is_sparse=False,
                   dynamic_axes=Axis.default_input_variable_dynamic_axes(), name=''):
    '''
    It creates an input in the network: a place where data,
    such as features and labels, should be provided.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (type, optional): np.float32 (default) or np.float64
        needs_gradients (bool, optional): whether to back-propagates to it or not. False by default.
        is_sparse (bool, optional): whether the variable is sparse (`False` by default)
        dynamic_axes (list or tuple, default): a list of dynamic axis (e.g., batch axis, time axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.variables.Variable`
    '''
    from cntk.cntk_py import input_variable
    from ..utils import sanitize_shape, sanitize_dtype_cntk

    shape = sanitize_shape(shape)

    if dtype is None:
        dtype = np.float32
    dtype = sanitize_dtype_cntk(dtype)
    dynamic_axes = sanitize_dynamic_axes(dynamic_axes)

    # TODO dynamic axis for numpy arrays
    # TODO sparse for numpy arrays

    return input_variable(shape, is_sparse, dtype, needs_gradient, name, dynamic_axes)


@typemap
def output_variable(shape, dtype, dynamic_axes, name=''):
    '''
    It creates an output node that is used to define a user defined function.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (type): np.float32 or np.float64
        dynamic_axes (list or tuple): a list of dynamic axis (e.g., batch axis, time axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.variables.Variable` that is of output type
    '''
    from cntk.cntk_py import output_variable
    from ..utils import sanitize_shape, sanitize_dtype_cntk

    shape = sanitize_shape(shape)

    dtype = sanitize_dtype_cntk(dtype)

    for a in dynamic_axes:
        if not a.is_dynamic_axis:
            raise ValueError('axis in dynamic_axes attribute is not dynamic')
    dynamic_axes = list(reversed(dynamic_axes))

    return output_variable(shape, dtype, dynamic_axes, name)


@typemap
def placeholder_variable(shape=None, dynamic_axes=None, name=''):
    '''
    It creates a variable place holder for recurrence networks, when the network's dynamic axes
    are unfolded, the place holder will get assigned a variable along the correspondent dynamic axis.

    Args:
        shape (tuple or int): the shape of the variable tensor
        dynamic_axes (list): the list of dynamic axes that the actual variable uses

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import placeholder_variable, NDShape, Axis

    if shape is None:
        shape = NDShape.unknown.dimensions()
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
        :class:`~cntk.ops.variables.Parameter`
    '''

    from .variables import Parameter
    if not device:
        device = use_default_device()

    if dtype is not None:
        if isinstance(init, np.ndarray) and dtype != init.dtype:
            init = np.array(init, dtype=dtype)
    else:
        if np.isscalar(init) and not shape:
            shape = ()
            if isinstance(init, np.ndarray):
                dtype = init.dtype
            else:
                dtype = np.float32
        else:
            dtype = None

    return Parameter(shape, init, dtype, device, name)


@typemap
def constant(value=None, shape=None, dtype=None, device=None, name=''):
    '''
    It creates a constant tensor initialized from a numpy array

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
        :class:`~cntk.ops.variables.Constant`
    '''
    from .variables import Constant
    if not device:
        device = use_default_device()
    #if np.isscalar(value) and not shape:
    if (np.isscalar(value) or isinstance(value, np.ndarray)) and not shape:
        shape = ()
    if dtype is not None:
        if isinstance(value, np.ndarray) and dtype != value.dtype:
            value = np.array(value, dtype=dtype)
    else:
        if isinstance(value, np.ndarray):
            dtype = value.dtype
        else:
            dtype = np.float32

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
