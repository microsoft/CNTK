# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk.internal import typemap, sanitize_input
from cntk.internal.utils import get_data_type

from ...axis import Axis
from ...default_options import get_default_override, default_override_or
##########################################################################
# variable ops
##########################################################################

@typemap
def input(shape, dtype=default_override_or(np.float32), needs_gradient=False, is_sparse=False,
          sequence_axis=Axis.default_dynamic_axis(), name=''):
    '''input(shape, dtype=np.float32, needs_gradient=False, is_sparse=False, sequence_axis=Axis.default_dynamic_axis(), name='')

    It creates an input in the network: a place where data,
    such as features and labels, should be provided.

    Args:
        shape (tuple or int): the shape of the input tensor
        dtype (np.float32 or np.float64): data type. Default is np.float32.
        needs_gradients (bool, optional): whether to back-propagates to it or not. False by default.
        is_sparse (bool, optional): whether the variable is sparse (`False` by default)
        dynamic_axes (list or tuple, default): a list of dynamic axis (e.g., batch axis, time axis)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.variables.Variable`
    '''
    from ... import input
    return input(shape, dtype, needs_gradient, is_sparse, [Axis.default_batch_axis(), sequence_axis], name)

##########################################################################
# sequence ops
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

    The initial state can be a constant (scalar or tensor), a learnable tensor
    or input data (which has a batch dimension, as needed for sequence-to-sequence models).

    Example:
        >>> x = C.sequence.input(shape=(3,2))
        >>> # Create one sequence with 4 tensors of shape (3, 2)
        >>> x0 = np.reshape(np.arange(24,dtype=np.float32),(1,4,3,2))
        >>> y = C.sequence.future_value(x) # using initial state of 0 by default
        >>> y.eval({x:x0})
        [array([[[  6.,   7.],
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
                 [  0.,   0.]]], dtype=float32)]

    Args:
        x: the tensor (or its name) from which the future value is obtained.
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the future (default 1)
        name (str, optional): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from cntk.internal import sanitize_dtype_cntk
    from ...cntk_py import Constant
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

    The initial state can be a constant (scalar or tensor), a learnable tensor
    or input data (which has a batch dimension, as needed for sequence-to-sequence models).

    Example:
        >>> # create example input: one sequence with 4 tensors of shape (3, 2)
        >>> from cntk.layers.typing import Tensor, Sequence
        >>> x = C.sequence.input((3,2))
        >>> x0 = np.reshape(np.arange(24,dtype=np.float32),(1,4,3,2))
        >>> x0
        array([[[[  0.,   1.],
                 [  2.,   3.],
                 [  4.,   5.]],
        <BLANKLINE>
                [[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]],
        <BLANKLINE>
                [[ 12.,  13.],
                 [ 14.,  15.],
                 [ 16.,  17.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]]], dtype=float32)

        >>> # this demonstrates how past_value shifts the sequence by one, padding with initial_state
        >>> y = C.sequence.past_value(x) # initial_state is 0 by default
        >>> y.eval({x:x0})
        [array([[[  0.,   0.],
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
                 [ 16.,  17.]]], dtype=float32)]

        >>> # here, we pass a the initial_state as input data (e.g. sequence-to-sequence)
        >>> s = C.input((3,2))  # not a sequence, e.g. a final encoder hidden state
        >>> s0 = np.reshape(np.arange(6,dtype=np.float32)/2,(1,3,2))
        >>> s0
        array([[[ 0. ,  0.5],
                [ 1. ,  1.5],
                [ 2. ,  2.5]]], dtype=float32)
        >>> y = C.sequence.past_value(x, initial_state=s)
        >>> y.eval({x:x0, s:s0}) # same as the previous example except for the first time step
        [array([[[  0. ,   0.5],
                 [  1. ,   1.5],
                 [  2. ,   2.5]],
        <BLANKLINE>
                [[  0. ,   1. ],
                 [  2. ,   3. ],
                 [  4. ,   5. ]],
        <BLANKLINE>
                [[  6. ,   7. ],
                 [  8. ,   9. ],
                 [ 10. ,  11. ]],
        <BLANKLINE>
                [[ 12. ,  13. ],
                 [ 14. ,  15. ],
                 [ 16. ,  17. ]]], dtype=float32)]

    Args:
        x: the tensor (or its name) from which the past value is obtained
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the past (default 1)
        name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    from cntk.internal import sanitize_dtype_cntk
    from cntk.cntk_py import Constant, past_value

    if initial_state is None:
        initial_state = Constant.scalar(sanitize_dtype_cntk(np.float32), 0.0)
    else:
        initial_state = sanitize_input(initial_state)

    x = sanitize_input(x)
    return past_value(x, initial_state, time_step, name)


def delay(x, initial_state=None, time_step=1, name=''):
    '''
    This function combines :func:`~cntk.ops.sequence.past_value` and :func:`~cntk.ops.sequence.future_value` into a single function.
    This is useful when the time_step is computed and can be positive, negative, or 0.

    Args:
        x: the tensor (or its name) from which the past value is obtained
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        time_step (int): the number of time steps to look into the past, where negative values mean to look into the future, and 0 means a no-op (default 1).
        name (str, optional): the name of the Function instance in the network
    '''
    from ...ops import alias, element_select, element_divide, placeholder, exp 
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
        >>> x = C.sequence.input(shape=(3,2))
        >>> y = C.sequence.is_first(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        [array([ 1.,  0.,  0.,  0.], dtype=float32)]

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> y = C.sequence.is_last(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        [array([ 0.,  0.,  0.,  1.], dtype=float32)]

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
        Indexing in NumPy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> y = C.sequence.first(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.]]], dtype=float32)

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> y = C.sequence.last(x)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        array([[[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]], dtype=float32)

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x), 60)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0, dtype=np.float32), (1,4,3,2))
        >>> z.eval({x:x0})
        [array([[ 0.],
                [ 0.],
                [ 1.],
                [ 1.]], dtype=float32)]
        >>> y = C.sequence.where(z)
        >>> y.eval({x:x0})
        [array([ 2.,  3.], dtype=float32)]

        >>> # repeat frame[1] twice, frame[3] three times, and frame[4] twice
        >>> C.sequence.where(C.sequence.input(1)).eval([[[1], [2], [1], [3], [2]]])
        [array([ 0.,  1.,  1.,  2.,  3.,  3.,  3.,  4.,  4.], dtype=float32)]
        >>> # note that the above are the indices that are passed to 

        >>> # repeat frames with a fractional factor
        >>> C.sequence.where(C.sequence.input(1)).eval([[[1.2]]*10])
        [array([ 0.,  0.,  1.,  2.,  3.,  4.,  5.,  5.,  6.,  7.,  8.,  9.],
            dtype=float32)]
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
        >>> x = C.sequence.input(shape=(3,2))
        >>> z = C.greater(C.reduce_sum(x),60)
        >>> y = C.sequence.gather(x,z)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        [array([[[ 12.,  13.],
                 [ 14.,  15.],
                 [ 16.,  17.]],
        <BLANKLINE>
                [[ 18.,  19.],
                 [ 20.,  21.],
                 [ 22.,  23.]]], dtype=float32)]

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.scatter(t, b)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        [array([[[ 18.,  19.],
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
                 [  0.,   0.]]], dtype=float32)]

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> t = C.sequence.last(x)
        >>> b = C.sequence.is_first(x)
        >>> y = C.sequence.broadcast_as(t, b)
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y.eval({x:x0})
        [array([[[ 18.,  19.],
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
                 [ 22.,  23.]]], dtype=float32)]

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
        >>> x = C.sequence.input(shape=(3,2))
        >>> # create one sequence of 4 tensors each with shape (3,2)
        >>> x0 = np.reshape(np.arange(24.0,dtype=np.float32),(1,4,3,2))
        >>> y = C.sequence.reduce_sum(x)
        >>> y.eval({x:x0})
        array([[[ 36.,  40.],
                 [ 44.,  48.],
                 [ 52.,  56.]]], dtype=float32)

    Args:
        seq: sequence input tensor
        name (`str`, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    from cntk.cntk_py import sequence_reduce_sum
    seq = sanitize_input(seq, get_data_type(seq))
    return sequence_reduce_sum(seq, name)

@typemap
def reduce_max(x, name=''):
  """
  Get the max value in the sequence values

  Args:
    x: input sequence
    name: the name of the operator
  Returns:
    The max value in the input sequence
  """
  from ...ops import element_select, placeholder, greater 
  m = placeholder(shape=(1,), dynamic_axes = x.dynamic_axes, name='max')
  o = element_select(greater(x, future_value(m)), x, future_value(m))
  rlt = o.replace_placeholders({m:sanitize_input(o)})
  max_out = first(rlt) 
  return sanitize_input(max_out)

@typemap
def softmax(x, name = ''):
  """
  Compute softmax along with a squence values
  """
  from ...ops import element_divide, exp 
  x_exp = exp((x-broadcast_as(reduce_max(x), x))*10)
  x_softmax = element_divide(x_exp, broadcast_as(reduce_sum(x_exp), x), name = name)
  return x_softmax
