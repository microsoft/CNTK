# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Loss functions.
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from ..ops.functions import CloneMethod, Function
from ..variables import Variable, Parameter, Constant
from cntk.internal import sanitize_input, sanitize_shape, sanitize_axis, sanitize_dynamic_axes, typemap
from cntk.internal.utils import get_data_type
from cntk.cntk_py import sentinel_value_for_auto_select_random_seed as auto_select
from cntk import times, softmax, parameter
import cntk as C
from ..axis import Axis

@typemap
def cosine_distance(x, y, name=''):
    '''
    Computes the cosine distance between ``x`` and ``y``:

    Example:
        >>> a = np.asarray([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]).reshape(3,2,2)
        >>> b = np.asarray([1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1]).reshape(3,2,2)
        >>> x = C.sequence.input_variable(shape=(2,))
        >>> y = C.sequence.input_variable(shape=(2,))
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
        >>> x = C.sequence.input_variable(shape=(4,))
        >>> y = C.sequence.input_variable(shape=(4,))
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

    Args:
        output: the computed posterior probability for a variable to be 1 from the network (typ. a ``sigmoid``)
        target: ground-truth label, 0 or 1
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Todo:
        add an example
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

    Args:
        output: the computed posterior probability from the network
        target: ground-truth label, 0 or 1
        weight: weight of each example
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`

    Todo:
        add an example
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
        >>> i1 = C.input_variable((1,2))
        >>> i2 = C.input_variable((1,2))
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
        >>> group = C.input_variable((1,))
        >>> score = C.input_variable((1,), needs_gradient=True)
        >>> gain  = C.input_variable((1,))
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


@typemap
def nce_loss(weights, biases, inputs, labels, noise_distribution, num_samples=32, allow_duplicates=True, seed=auto_select, name=''):
    '''nce_loss(weights, biases, inputs, labels, noise_distribution, num_samples=32, allow_duplicates=True, seed=auto_select, name='')
    Computes the noise contrastive estimation loss. This implementation mostly
    follows Chris Dyer's notes [1]. At a high level, this layer draws
    `num_samples` random labels from `noise_distribution` and then forms
    `num_samples`+1 binary classification problems where the true label is
    considered a positive example and the random labels are considered negative
    examples. The negatives are shared among all the examples in the
    minibatch. This operation only computes the logits for the labels in the
    minibatch and the random labels drawn from `noise_distribution`. The
    gradients will be sparse if the labels are sparse.

    The `noise_distribution` is read once and certain quantities are
    precomputed based on it. This operation will need to be reinstantiated if
    the `noise_distribution` changes.

    Shape inference for the weights is currently not supported when inputs are
    placeholders. Either a concrete input must be used or the weights must be
    provided without any inferred dimensions.

    Example:
        >>> import scipy
        >>> # dimensions of input, number of noise labels, batch size, number of classes
        >>> xdim = 10
        >>> samples = 32
        >>> batch = 4
        >>> classes = 100
        >>> # some variables; typically x will be the output of a layer
        >>> x = C.input_variable(xdim)
        >>> y = C.input_variable(classes, is_sparse=True)
        >>> # dummy data
        >>> x0 = np.arange(batch * xdim, dtype=np.float32).reshape((batch, xdim))/(batch * xdim)
        >>> data = np.ones(batch, dtype=np.float32)
        >>> indices = list(range(10, 10*batch+1, 10))
        >>> indptr = list(range(batch + 1))
        >>> y0 = scipy.sparse.csr_matrix((data, indices, indptr), shape=(batch, classes))
        >>> # a dummy noise distribution
        >>> q = np.arange(classes, dtype=np.float32) + 1 # normalization not necessary
        >>> # the parameters
        >>> b = C.parameter((classes, 1), init=-np.log(classes))
        >>> W = C.parameter((classes, C.InferredDimension), init=C.glorot_uniform(seed=98052))
        >>> # the loss
        >>> loss = C.nce_loss(W, b, x, y, q, seed=98052)
        >>> # evaluate the loss at our dummy data
        >>> np.round(loss.eval({x:x0, y:y0}), decimals=3)
        array([ 2.385,  3.035,  3.886,  3.868], dtype=float32)
        >>> # after training, use the logits for predictions
        >>> logits = C.times_transpose(x, W) + C.reshape(b, -1)

    Args:
        weights: parameter (or variable in general) containing the weights with
         which inputs will be multiplied. Its shape must be
         (number of classes, dimension of input)
        biases: parameter (or variable in general) containing the biases that
         will be added to the product of weights and inputs. Its shape must be
         (number of classes, 1)
        inputs: vector of inputs to this layer. Multiplying by the weights and
         adding the biases gives the logits.
        labels: a one-hot vector with the ground-truth labels.
        noise_distribution: a constant vector with dimension equal to the number
         of classes. The entries must be positive numbers but do not have to
         sum to 1. random labels will be drawn according to the normalized
         distribution.
        num_samples: number of random labels that will be drawn from the
         `noise_distribution`.
        allow_duplicates: boolean. If True (default), the random labels can
         contain duplicates. Compared to `allow_duplicates=False` it is faster
         but the quality of the approximations is slightly worse for the same
         number of samples.
        seed: random seed. The default value selects a unique random seed.
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`

    See also:
        [1] C. Dyer. `Notes on Noise Contrastive Estimation and Negative Sampling [pdf] <http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf>`_.
    '''
    from cntk.cntk_py import nce_loss
    dtype = get_data_type(inputs, labels, noise_distribution)
    inputs = sanitize_input(inputs, dtype)
    labels = sanitize_input(labels, dtype)
    noise_distribution = sanitize_input(noise_distribution, dtype)
    return nce_loss(weights, biases, inputs, labels, noise_distribution,
                    num_samples, allow_duplicates, seed, name)

@typemap
def lattice_sequence_with_softmax(label, prediction, loglikelihood, lattice, symListPath, phonePath, stateListPath, transProbPath, latticeConfigPath="LatticeNode.config", 
                                  hSmoothingWeight = 0.95, frameDropThresh = 1e-10, doReferenceAlign = False, seqGammarUsesMBR = False, 
                                  seqGammarAMF = 14.0, seqGammarLMF = 14.0, seqGammarBMMIFactor = 0.0, seqGammarWordPen = 0.0, name=''):
    from cntk.cntk_py import lattice_sequence_with_softmax
    dtype = get_data_type(label, prediction, loglikelihood, lattice)
    label = sanitize_input(label, dtype)
    prediction = sanitize_input(prediction, dtype)
    loglikelihood = sanitize_input(loglikelihood, dtype)
    lattice = sanitize_input(lattice, dtype)
    return lattice_sequence_with_softmax(label, prediction, loglikelihood, lattice, symListPath, phonePath, stateListPath, transProbPath, latticeConfigPath, hSmoothingWeight, frameDropThresh, doReferenceAlign, seqGammarUsesMBR, seqGammarAMF, seqGammarLMF, seqGammarBMMIFactor, seqGammarWordPen, name)

@typemap
def hierarchical_softmax_layer(input_var, label_index, label_dim, label_classes=None):
    '''
    A two layers hierarchical softmax function:

    Args:
        input_var: Variable with shape: [#,*](dim_x)
        label_index: index of label's category:  [#,*](1)
        label_dim: number of the label categories
        label_classes: number of classes of the label categories
    Returns:
        output_prob: the probability of the given label [#,*](1)
        class_probs: the probability of all the label classes [#,*](label_classes)
        all_probs: the probability of all label classes 
    '''
    input_dim = input_var.shape[0]

    if not label_classes:
        label_classes = int(np.ceil(np.sqrt(float(label_dim))))

    n_outputs_per_class = int(np.ceil(label_dim / label_classes))

    target_class = C.floor((label_index + 0.5) / n_outputs_per_class)
    target_output_in_class = C.round(label_index - target_class * n_outputs_per_class)

    w1 = parameter(shape=(input_dim, label_classes), init=C.glorot_normal(), name='hsoftmax_w1')
    b1 = parameter(shape=(label_classes), init=C.glorot_normal(), name='hsoftmax_b1')
    w2s = parameter(shape=(label_classes, input_dim, n_outputs_per_class,), init=C.glorot_normal(), name='hsoftmax_w2s')
    b2s = parameter(shape=(label_classes, n_outputs_per_class,), init=C.glorot_normal(), name='hsoftmax_b2s')

    class_probs = softmax(b1 + times(input_var, w1))

    # TODO: fix the bug in backprop for sparse, and use sparse embedding to accelerate
    target_class_one_hot = C.one_hot(target_class, num_classes=label_classes, sparse_output=False)
    w2 = C.reshape(C.times(target_class_one_hot, w2s, output_rank=2), [input_dim, -1])
    b2 = C.reshape(times(target_class_one_hot, b2s, output_rank=1), [-1])
    probs_in_class = softmax(b2 + times(input_var, w2))

    prob_in_class = C.times_transpose(C.one_hot(target_output_in_class, num_classes=n_outputs_per_class, sparse_output=False), probs_in_class)
    class_prob = C.times_transpose(C.one_hot(target_class, num_classes=label_classes, sparse_output=False), class_probs)
    output_prob = prob_in_class * class_prob

    # this is for calculating all the outputs' probabilities
    all_probs = []
    for i in range(label_classes):
        ci = C.constant(i)
        ci_one_hot = C.one_hot(ci, num_classes=label_classes, sparse_output=False)
        w2a = C.times(ci_one_hot, w2s, output_rank=2)
        b2a = C.times(ci_one_hot, b2s, output_rank=1)
        probs_in_classa = C.softmax(b2a + times(input_var, w2a))
        class_proba = C.times_transpose(ci_one_hot, class_probs)
        output_proba = probs_in_classa * class_proba
        all_probs.append(output_proba)

    return output_prob, class_probs, all_probs


@typemap
def fmeasure(output, target, beta=1):
    """
    This operation computes the f-measure between the output and target. If beta is set as one,
    its called the f1-scorce or dice similarity coefficient. f1-scorce is monotonic in jaccard distance.

    f-measure = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    This loss function is frequently used in semantic segmentation of images. Works with imbalanced classes, for
    balanced classes you should prefer cross_entropy instead.
    This operation works with both binary and multiclass classification.

    Args:
        output: the output values from the network
        target: it is usually a one-hot vector where the hot bit corresponds to the label index
        beta: greater than one weights recall higher than precision, less than one for the opposite.
        Commonly chosen values are 0.5, 1 or 2.

    Returns:
        :class:`~cntk.ops.functions.Function`

    """

    assert len(target.shape) == len(output.shape)

    if len(output.shape) == 3:
        axis = (1, 2)  # assumes that the first axis is the class axis
    else:
        axis = None

    correct_predictions = C.reduce_sum(output * target, axis=axis)
    precision = correct_predictions / C.reduce_sum(output, axis=axis)
    recall = correct_predictions / C.reduce_sum(target, axis=axis)
    return 1 - (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
