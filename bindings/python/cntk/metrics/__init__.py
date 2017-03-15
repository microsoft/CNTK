# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from __future__ import print_function
import numpy as np
from ..ops.functions import CloneMethod, Function, load_model
from ..ops.variables import Variable, Parameter, Constant
from ..utils import get_data_type
from cntk.internal import sanitize_input, sanitize_shape, sanitize_axis, sanitize_dynamic_axes, typemap
from ..axis import Axis

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
        >>> np.round(f.grad({score:s, gain:n, group: g}, wrt=[score]),4)
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
def edit_distance_error(input_a, input_b, subPen=0, delPen=0, insPen=0, squashInputs=False, tokensToIgnore=[], name=''):
    '''
    Edit distance error evaluation node with the option of specifying penalty of substitution, deletion and insertion, as well as squashing the input sequences and ignoring certain samples.
    Using the classic DP algorithm as described in https://en.wikipedia.org/wiki/Edit_distance, adjusted to take into account the penalties.

    Each sequence in the inputs is expected to be a matrix. Prior to computation of the edit distance, the operation extracts the indices of maximum element in each column.
    For example, a sequence matrix
    1 2 9 1
    3 0 3 2
    will be represented as the vector of labels (indices) as [1, 0, 0, 1], on which edit distance will be actually evaluated.

    The node allows to squash sequences of repeating labels and ignore certain labels. For example, if squashInputs is true and tokensToIgnore contains label '-' then
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
        squashInputs: whether to merge sequences of identical samples (in both input sequences). If true and tokensToIgnore contains label '-' then
                given first input sequence as s1="a-ab-" and second as s2="-aa--abb" the edit distance will be computed against s1' = "aab" and s2' = "aab".
        tokensToIgnore: list of samples to ignore during edit distance evaluation (in both sequences)
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import edit_distance_error
    dtype = get_data_type(input_a, input_b)
    input_a = sanitize_input(input_a, dtype)
    input_b = sanitize_input(input_b, dtype)
    return edit_distance_error(input_a, input_b, subPen, delPen, insPen, squashInputs, tokensToIgnore, name)
