# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


from __future__ import division
from __future__ import print_function
import numpy as np
from ..ops.functions import CloneMethod, Function, load_model
from ..variables import Variable, Parameter, Constant
from cntk.internal import sanitize_input, sanitize_shape, sanitize_axis, sanitize_dynamic_axes, typemap
from cntk.internal.utils import get_data_type
from ..axis import Axis


@typemap
def cosine_distance(x, y, name=''):
    '''
    Computes the cosine distance between ``x`` and ``y``:

    Example:
        >>> a = np.asarray([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]).reshape(3,2,2)
        >>> b = np.asarray([1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1]).reshape(3,2,2)
        >>> x = C.sequence.input(shape=(2,))
        >>> y = C.sequence.input(shape=(2,))
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

# TODO: Per discussion with sayanp, the underlying C++ code is not fully functional, so this
#       should be marked as deprecated (a final design would separate negative sampling and cosine distance).
@typemap
def cosine_distance_with_negative_samples(x, y, shift, num_negative_samples, name=''):
    '''

    Given minibatches for ``x`` and ``y``, this function computes for each element in `x` the cosine distance between 
    it and the corresponding `y` and additionally the cosine distance between ``x`` and some other elements of ``y`` 
    (referred to a negative samples). The ``x`` and ``y`` pairs are samples often derived 
    from embeddings of textual data, though the function can be used for any form of numeric encodings. 
    When using this function to compute textual similarity, ``x`` represents search query term embedding 
    and ``y`` represents a document embedding. The negative samples are formed on the fly by shifting 
    the right side (``y``). The ``shift`` indicates how many samples in ``y`` one should shift while
    forming each negative sample pair. It is often chosen to be 1. As the name suggests 
    ``num_negative_samples`` indicates how many negative samples one would want to generate.

    Example:
        >>> qry = np.asarray([1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1.], dtype=np.float32).reshape(3, 1, 4)
        >>> doc = np.asarray([1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1.], dtype=np.float32).reshape(3, 1, 4)
        >>> x = C.sequence.input(shape=(4,))
        >>> y = C.sequence.input(shape=(4,))
        >>> model = C.cosine_distance_with_negative_samples(x, y, shift=1, num_negative_samples=2)
        >>> np.round(model.eval({x: qry, y: doc}), decimals=4)
        array([[[ 1. ,  0.5,  0. ]],
        <BLANKLINE>
               [[ 1. ,  0.5,  0.5]],
        <BLANKLINE>
               [[ 1. ,  0. ,  0.5]]], dtype=float32)

    Args:
        x: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        y: numpy array or any :class:`~cntk.ops.functions.Function` that outputs a tensor
        shift: non-zero positive integer representing number of shift to generate a negative sample
        num_negative_samples: number of negative samples to generate, a non-zero positive integer 
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import cosine_distance_with_negative_samples
    dtype = get_data_type(x, y)
    x = sanitize_input(x, dtype)
    y = sanitize_input(y, dtype)

    return cosine_distance_with_negative_samples(x, y, shift, num_negative_samples, name)

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
        axis (int or :class:`~cntk.axis.Axis`, optional): if given, cross entropy will be computed
                along this axis
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
        >>> i1 = C.input((1,2))
        >>> i2 = C.input((1,2))
        >>> C.squared_error(i1,i2).eval({i1:np.asarray([[[2., 1.]]], dtype=np.float32), i2:np.asarray([[[4., 6.]]], dtype=np.float32)})
        array([ 29.], dtype=float32)

        >>> C.squared_error(i1,i2).eval({i1:np.asarray([[[1., 2.]]], dtype=np.float32), i2:np.asarray([[[1., 2.]]], dtype=np.float32)})
        array([ 0.], dtype=float32)

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
        >>> group = C.input((1,))
        >>> score = C.input((1,), needs_gradient=True)
        >>> gain  = C.input((1,))
        >>> g = np.array([1, 1, 2, 2], dtype=np.float32).reshape(4,1)
        >>> s = np.array([1, 2, 3, 4], dtype=np.float32).reshape(4,1)
        >>> n = np.array([7, 1, 3, 1], dtype=np.float32).reshape(4,1)
        >>> f = C.lambda_rank(score, gain, group)
        >>> np.round(f.grad({score:s, gain:n, group: g}, wrt=[score]),4)
        array([[-0.2121],
        <BLANKLINE>
               [ 0.2121],
        <BLANKLINE>
               [-0.1486],
        <BLANKLINE>
               [ 0.1486]], dtype=float32)

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
