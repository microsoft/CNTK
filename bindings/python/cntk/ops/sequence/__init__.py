# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ...utils import sanitize_input, sanitize_shape, get_data_type, typemap

##########################################################################
# sequence ops
##########################################################################


@typemap
def is_first(seq, name=''):
    '''
    Returns a symbolic sequence of booleans with the same length as ``seq``. The
    first element of the sequence is 1 and all others are 0.

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.is_first(x)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[ 1.,  0.,  0.,  0.]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import is_first
    seq = sanitize_input(seq, get_data_type(seq))
    return is_first(seq, name)


@typemap
def is_last(seq, name=''):
    '''
    Returns a symbolic sequence of booleans with the same length as ``seq``. The
    last element of the sequence is 1 and all others are 0.

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.is_last(x)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[ 0.,  0.,  0.,  1.]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.Function`:
    '''
    from cntk.cntk_py import is_last
    seq = sanitize_input(seq, get_data_type(seq))
    return is_last(seq, name)


@typemap
def first(seq, name=''):
    '''
    Returns the first element of its symbolic input sequence ``seq``

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.first(x)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.]]]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import first
    seq = sanitize_input(seq, get_data_type(seq))
    return first(seq, name)


@typemap
def last(seq, name=''):
    '''
    Returns the last element of its symbolic input sequence ``seq``

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.last(x)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import last
    seq = sanitize_input(seq, get_data_type(seq))
    return last(seq, name)


@typemap
def where(condition, name=''):
    '''
    Given a symbolic sequence ``condition`` of boolean-like values, it will return
    a new sequence containing the indices for which the values were true.

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x),60)
        >>> y = C.sequence.where(z)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> z.eval({x:x0})
        array([[ 0.,  0.,  1.,  1.]], dtype=float32)
        >>> y.eval({x:x0})
        array([[ 2.,  3.]], dtype=float32)

    Args:
        condition: the symbolic sequence of booleans
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import where
    condition = sanitize_input(condition, get_data_type(condition))
    return where(condition, name)

@typemap
def gather(seq, condition, name=''):
    '''
    Takes two sequences of the same length and returns a new sequence whose
    elements are those elements of sequence ``seq`` whose corresponding element
    in ``condition`` is True, preserving the ordering of ``seq``.

    This operation is also known as stream compaction, or copy_if.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x),60)
        >>> y = C.sequence.gather(x,z)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 12.,  13.],
                 [ 14.,  15.],
                 [ 16.,  17.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]]], dtype=float32)

    Args:
        seq: the symbolic sequence from which elements will be selected
        condition: the symbolic sequence of booleans which indicate which
            elements should be selected
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import gather
    seq = sanitize_input(seq, get_data_type(seq))
    condition = sanitize_input(condition, get_data_type(condition))
    return gather(seq, condition, name)


@typemap
def scatter(seq, condition, name=''):
    '''
    Performs the inverse of gather. The sequence ``seq`` must have as many
    elements as the number of True values in the sequence ``condition``.
    It will return a sequence whose length is the same as the ``condition``
    sequence with zeroes everywhere except for the locations where ``condition``
    evaluates to True in which case it will copy the elements from ``seq``
    preserving their order.

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.scatter(t, b)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]],
        <BLANKLINE>
                [[  0.,   0.],
                 [  0.,   0.],
                 [  0.,   0.]],
        <BLANKLINE>
                [[  0.,   0.],
                 [  0.,   0.],
                 [  0.,   0.]],
        <BLANKLINE>
                [[  0.,   0.],
                 [  0.,   0.],
                 [  0.,   0.]]]], dtype=float32)

    Args:
        seq: the symbolic sequence from which elements will be copied in the
            output
        condition: the symbolic sequence which denotes the locations where
            elements should be copied
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import scatter
    seq = sanitize_input(seq, get_data_type(seq))
    condition = sanitize_input(condition, get_data_type(condition))
    return scatter(seq, condition, name)


@typemap
def broadcast_as(operand, broadcast_as_operand, name=''):
    '''
    Creates a sequence out of a non-sequence by endowing the ``operand``
    with dynamic axes of the same type as the ``broadcast_as_operand``
    and broadcasting the value of the ``operand`` along those dynamic axes.

    Example:
        >>> import cntk.ops as C
        >>> import numpy as np
        >>> x = C.input_variable(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.broadcast_as(t, b)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]]], dtype=float32)

    Args:
        operand: the symbolic tensor whose value will be broadcast
        broadcast_as_operand: the symbolic tensor whose dynamic axes will
            be used to broadcast the operand
        name (str): the name of the node in the network

    Returns:
        :class:`cntk.Function`
    '''
    from cntk.cntk_py import broadcast_as
    operand = sanitize_input(operand, get_data_type(operand))
    broadcast_as_operand = sanitize_input(
        broadcast_as_operand, get_data_type(broadcast_as_operand))
    return broadcast_as(operand, broadcast_as_operand, name)
