# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ...utils import sanitize_input, get_data_type, typemap

##########################################################################
# sequence ops
##########################################################################

def delay(x, initial_state=None, time_step=1, name=''):
    '''
    This function combines ``past_value`` and ``future_value`` into a single function.
    This is useful when the time_step can be specified as positive or negative.

    Args:
        x: the tensor (or its name) from which the past value is obtained
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the past, where negative values mean to look into the future, and 0 means a no-op (default 1).
        name (str, optional): the name of the Function instance in the network
    '''
    from ...ops import alias, past_value, future_value
    if time_step > 0:
        return past_value  (x, time_step= time_step, initial_state=initial_state, name=name)
    elif time_step < 0:
        return future_value(x, time_step=-time_step, initial_state=initial_state, name=name)
    else:
        if name:
            return alias(x, name)
        else:
            return x


@typemap
def is_first(seq, name=''):
    '''
    Returns a symbolic sequence of booleans with the same length as ``seq``. The
    first element of the sequence is 1 and all others are 0.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.is_first(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[ 1.,  0.,  0.,  0.]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
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
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.is_last(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[ 0.,  0.,  0.,  1.]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import is_last
    seq = sanitize_input(seq, get_data_type(seq))
    return is_last(seq, name)

@typemap
def slice(seq, begin_index, end_index, name=''):
    '''
    Slice the input sequence.

    Examples:
        TBA
    Args:
        seq: sequence input tensor
        begin_index (`int`): the index along sequence axis where the slicing starts
        end_index (`int`): the index along sequence axis where the slicing ends
        name (`str`, optional): the name of the Function instance in the network

    See also:
        Indexing in NumPy: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sequence_slice
    seq = sanitize_input(seq, get_data_type(seq))
    return sequence_slice(seq, begin_index, end_index, name)

@typemap
def first(seq, name=''):
    '''
    Returns the first element of its symbolic input sequence ``seq``

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.first(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.]]]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import first
    seq = sanitize_input(seq, get_data_type(seq))
    return first(seq, name)


@typemap
def last(seq, name=''):
    '''
    Returns the last element of its symbolic input sequence ``seq``

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> y = C.sequence.last(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[[[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]]], dtype=float32)

    Args:
        seq: the symbolic tensor denoting a sequence
        name (str): the name of the node in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import last
    seq = sanitize_input(seq, get_data_type(seq))
    return last(seq, name)


@typemap
def where(condition, name=''):
    '''
    Given a symbolic sequence ``condition`` of boolean-like (1/0) values, it will return
    a new sequence containing the indices for which the values were true.

    If ``condition`` has a value other than 0 or 1, it will denote a repeat factor.
    If a repeat factor is fractional, it will round up but deduct the overshoot from the
    next repeat factor.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x), 60)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0, dtype=np.float32), (1,4,3,2))
        >>> z.eval({x:x0})
        array([[[ 0.],
                [ 0.],
                [ 1.],
                [ 1.]]], dtype=float32)
        >>> y = C.sequence.where(z)
        >>> y.eval({x:x0})
        array([[ 2.,  3.]], dtype=float32)

        >>> # repeat frame[1] twice, frame[3] three times, and frame[4] twice
        >>> C.sequence.where(C.input_variable(1)).eval([[[1], [2], [1], [3], [2]]])
        array([[ 0.,  1.,  1.,  2.,  3.,  3.,  3.,  4.,  4.]], dtype=float32)
        >>> # note that the above are the indices that are passed to 

        >>> # repeat frames with a fractional factor
        >>> C.sequence.where(C.input_variable(1)).eval([[[1.2]]*10])
        array([[ 0.,  0.,  1.,  2.,  3.,  4.,  5.,  5.,  6.,  7.,  8.,  9.]], dtype=float32)
        >>> # as a result, a 1.2 times stretch is realized by duplicating frame[0] and frame[5]

    Args:
        condition: sequence of 0 or 1 values for filtering, or other positive values for repetition (also fractional)
        name (str): the name of the node in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import where
    condition = sanitize_input(condition, get_data_type(condition))
    return where(condition, name)

@typemap
def gather(seq, condition, new_sequence_axis_typeinfo=None, name=''):
    '''
    Takes two sequences of the same length and returns a new sequence whose
    elements are those elements of sequence ``seq`` whose corresponding element
    in ``condition`` is True, preserving the ordering of ``seq``.

    This operation is also known as stream compaction, or copy_if.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x),60)
        >>> y = C.sequence.gather(x,z)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
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
        new_sequence_axis_typeinfo:  tuple of integers indicating
            the scaling and additive factors for the length of the new sequence axis
            w.r.t. the operand sequence. This is used to determine the sequence axis
            to be used for the output of the gather operation. If this argument is left 
            unspecified, a new independent sequence axis is created.
        name (str): the name of the node in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import gather
    seq = sanitize_input(seq, get_data_type(seq))
    condition = sanitize_input(condition, get_data_type(condition))
    if new_sequence_axis_typeinfo is None:
        return gather(seq, condition, name)
    else:
        return gather(seq, condition, new_sequence_axis_typeinfo, name)


@typemap
def scatter(seq, condition, new_sequence_axis_typeinfo=None, name=''):
    '''
    Performs the inverse of gather. The sequence ``seq`` must have as many
    elements as the number of True values in the sequence ``condition``.
    It will return a sequence whose length is the same as the ``condition``
    sequence with zeroes everywhere except for the locations where ``condition``
    evaluates to True in which case it will copy the elements from ``seq``
    preserving their order.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.scatter(t, b)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
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
        new_sequence_axis_typeinfo:  tuple of integers indicating
            the scaling and additive factors for the length of the new sequence axis
            w.r.t. the condition sequence. This is used to determine the sequence axis
            to be used for the output of the gather operation. If this argument is left 
            unspecified a new independent sequence axis is created.
        name (str): the name of the node in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import scatter
    seq = sanitize_input(seq, get_data_type(seq))
    condition = sanitize_input(condition, get_data_type(condition))
    if new_sequence_axis_typeinfo is None:
        return scatter(seq, condition, name)
    else:
        return scatter(seq, condition, new_sequence_axis_typeinfo, name)


@typemap
def broadcast_as(operand, broadcast_as_operand, name=''):
    '''
    Creates a sequence out of a non-sequence by endowing the ``operand``
    with dynamic axes of the same type as the ``broadcast_as_operand``
    and broadcasting the value of the ``operand`` along those dynamic axes.

    Example:
        >>> x = C.input_variable(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.broadcast_as(t, b)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
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
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import broadcast_as
    operand = sanitize_input(operand, get_data_type(operand))
    broadcast_as_operand = sanitize_input(
        broadcast_as_operand, get_data_type(broadcast_as_operand))
    return broadcast_as(operand, broadcast_as_operand, name)

@typemap
def reduce_sum(seq, name=''):
    '''
    Computes the sum of the input sequence's elements across the sequence axis.

    Examples:
        >>> x = C.input_variable(shape=(3,2))
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y = C.sequence.reduce_sum(x)
        >>> y.eval({x:x0})
        array([[[[ 36.,  40.],
                 [ 44.,  48.],
                 [ 52.,  56.]]]], dtype=float32)

    Args:
        seq: sequence input tensor
        name (`str`, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sequence_reduce_sum
    seq = sanitize_input(seq, get_data_type(seq))
    return sequence_reduce_sum(seq, name)
