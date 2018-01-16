# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import os
import numpy as np
import cntk as C
from cntk.tests.test_utils import *
from cntk.device import cpu, gpu
from ...ops.functions import Function
from cntk.internal import sanitize_dtype_cntk
from cntk.internal.utils import eval as cntk_eval
from .. import constant


def cntk_device(device_id):
    '''
    Converts the legacy device ID as it was used in CNTK 1 to a :class:`~cntk.device.DeviceDescriptor` instance.

    Args:
        device_id (int): device id, -1 for CPU, 0 or higher for GPU

    Returns:
        :class:`~cntk.device.DeviceDescriptor`
    '''
    if device_id == -1:
        return cpu()
    else:
        return gpu(device_id)


def os_process():
    '''
    Returns the process instance, which can be used e.g. to check the memory
    usage.
    '''
    import psutil
    return psutil.Process(os.getpid())


def mem_used(process):
    '''
    Return the non-swapped physical memory the Python process is using.
    '''
    return process.memory_info().rss


def _test_unary_op(precision, device_id, op_func,
                   value, expected_forward, expected_backward_all, op_param_dict={}):

    value = AA(value, dtype=PRECISION_TO_TYPE[precision])

    a = C.input_variable(shape=value.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

    # create batch
    value.shape = (1,) + value.shape

    if (type(op_func) == str):
        input_op = eval('%s a' % op_func)
    else:
        input_op = op_func(a, **op_param_dict)

    forward_input = {a: value}
    expected_backward = {a: expected_backward_all['arg'], } if expected_backward_all is not None else None
    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def _test_binary_op(precision, device_id, op_func, left_operand, right_operand,
                    expected_forward, expected_backward_all, wrap_batch_seq=True, op_param_dict={}, batch_size_greater_than_one=False):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    left_value = AA(left_operand, dtype=dt)
    right_value = AA(right_operand, dtype=dt)
    
    left_operand_shape = left_value.shape[1:] if batch_size_greater_than_one else left_value.shape
    right_operand_shape = right_value.shape[1:] if batch_size_greater_than_one else right_value.shape

    a = C.input_variable(shape=left_operand_shape,
              dtype=sanitize_dtype_cntk(precision),
              needs_gradient=True,
              name='a')

    b = C.input_variable(shape=right_operand_shape,
              dtype=sanitize_dtype_cntk(precision),
              needs_gradient=True,
              name='b')

    const_a = constant(left_value, device=dev)
    const_b = constant(right_value, device=dev)

    if (type(op_func) == str):
        input_op_constant = eval('a %s const_b' % op_func)
        constant_op_input = eval('const_a %s b' % op_func)
        input_op_input = eval('a %s b' % op_func)
    else:
        input_op_constant = op_func(a, const_b, **op_param_dict)
        constant_op_input = op_func(const_a, b, **op_param_dict)
        input_op_input = op_func(a, b, **op_param_dict)

    # create batch by wrapping the data point into a batch of one sample
    if wrap_batch_seq and not batch_size_greater_than_one:
        left_value.shape = (1,) + left_value.shape
        right_value.shape = (1,) + right_value.shape

    forward_input = {a: left_value, b: right_value}
    expected_backward = {a: expected_backward_all[
        'left_arg'], b: expected_backward_all['right_arg'], }
    unittest_helper(input_op_input,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)
    
    if not batch_size_greater_than_one:
        forward_input = {a: left_value}
        expected_backward = {a: expected_backward_all['left_arg'], }
        unittest_helper(input_op_constant,
                        forward_input, expected_forward, expected_backward,
                        device_id=device_id, precision=precision)

        forward_input = {b: right_value}
        expected_backward = {b: expected_backward_all['right_arg'], }
        unittest_helper(constant_op_input,
                        forward_input, expected_forward, expected_backward,
                        device_id=device_id, precision=precision)


def unittest_helper(root_node,
                    forward_input, expected_forward, expected_backward,
                    device_id=-1, precision="float"):

    assert isinstance(root_node, Function)

    backward_pass = expected_backward is not None
    forward, backward = cntk_eval(root_node, forward_input, precision,
            cntk_device(device_id), backward_pass, expected_backward)

    # for forward we always expect only one result
    assert len(forward) == 1
    forward = list(forward.values())[0]

    forward = np.atleast_1d(forward)

    for res, exp in zip(forward, expected_forward):
        assert res.shape == AA(exp).shape
        assert np.allclose(res, exp, atol=TOLERANCE_ABSOLUTE)

    if expected_backward:
        for key in expected_backward:
            res, exp = backward[key], expected_backward[key]
            if isinstance(res, list):
                assert len(res) == len(exp)
                for res_seq, exp_seq in zip(res, exp):
                    assert res_seq.shape == AA(exp_seq).shape
                    assert np.allclose(
                        res_seq, exp_seq, atol=TOLERANCE_ABSOLUTE)

            elif isinstance(res, np.ndarray):
                assert res.shape == AA(exp).shape
                assert np.allclose(res, exp, atol=TOLERANCE_ABSOLUTE)


def batch_dense_to_sparse(batch, dynamic_axis=''):
    '''
    Helper test function that converts a batch of dense tensors into sparse
    representation that can be consumed by :func:`cntk.ops.sparse_input_numpy`.

    Args:
        batch (list): list of samples. If ``dynamic_axis`` is given, samples are sequences
         of tensors. Otherwise, they are simple tensors.
        dynamic_axis (str or :func:`cntk.ops.dynamic_axis` instance): the dynamic axis

    Returns:
        (indices, values, shape)
    '''

    batch_indices = []
    batch_values = []

    shapes_in_tensor = set()

    for tensor in batch:
        if isinstance(tensor, list):
            tensor = np.asarray(tensor)

        if dynamic_axis:
            # collecting the shapes ignoring the dynamic axis
            shapes_in_tensor.add(tensor.shape[1:])
        else:
            shapes_in_tensor.add(tensor.shape)

        if len(shapes_in_tensor) != 1:
            raise ValueError('except for the sequence dimensions all shapes ' +
                             'should be the same - instead we %s' %
                             (", ".join(str(s) for s in shapes_in_tensor)))

        t_indices = range(tensor.size)
        t_values = tensor.ravel(order='F')
        mask = t_values != 0

        batch_indices.append(list(np.asarray(t_indices)[mask]))
        batch_values.append(list(np.asarray(t_values)[mask]))

    return batch_indices, batch_values, shapes_in_tensor.pop()


def test_batch_dense_to_sparse_full():
    i, v, s = batch_dense_to_sparse(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[10, 20, 30], [40, 50, 60]],
        ])
    assert i == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
    ]
    assert v == [
        [1, 4, 2, 5, 3, 6],
        [10, 40, 20, 50, 30, 60]
    ]
    assert s == (2, 3)

    i, v, s = batch_dense_to_sparse([[1]])
    assert i == [[0]]
    assert v == [[1]]
    assert s == (1,)


def test_batch_dense_to_sparse_zeros():
    i, v, s = batch_dense_to_sparse(
        [
            [[1, 2, 3], [4, 0, 6]],
            [[0, 0, 0], [40, 50, 60]],
        ])
    assert i == [
        [0, 1, 2, 4, 5],
        [1, 3, 5],
    ]
    assert v == [
        [1, 4, 2, 3, 6],
        [40, 50, 60]
    ]
    assert s == (2, 3)

def remove_np_array_in_list(arr, l):
    index = 0
    size = len(l)
    while index != size and not np.allclose(l[index], arr, atol=TOLERANCE_ABSOLUTE):
        index += 1
    if index != size:
        l.pop(index)
    else:
        raise ValueError('array not found in list.')

# compare two unordered lists of np arrays
def compare_lists_of_np_arrays(first_list, second_list):
    second_list = list(second_list)   # make a mutable copy
    try:
        for elem in first_list:
            remove_np_array_in_list(elem, second_list)
    except ValueError:
        return False
    return not second_list
