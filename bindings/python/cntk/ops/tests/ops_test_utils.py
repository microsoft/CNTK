# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import numpy as np
import pytest

from cntk.tests.test_utils import *

from ...ops.functions import Function
from ...utils import sanitize_dtype_cntk
from ...utils import eval as cntk_eval, cntk_device
from .. import constant, input_variable

I = input_variable


@pytest.fixture(params=["dense", "sparse"])
def left_matrix_type(request):
    return request.param


@pytest.fixture(params=["dense", "sparse"])
def right_matrix_type(request):
    return request.param


def _test_unary_op(precision, device_id, op_func,
                   value, expected_forward, expected_backward_all, op_param_dict=None):

    value = AA(value, dtype=PRECISION_TO_TYPE[precision])

    a = I(shape=value.shape,
          data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    # create batch
    value.shape = (1, 1) + value.shape

    if (type(op_func) == str):
        input_op = eval('%s a' % op_func)
    elif op_param_dict:
        input_op = op_func(a, **op_param_dict)
    else:
        input_op = op_func(a)

    forward_input = {a: value}
    expected_backward = {a: expected_backward_all['arg'], }
    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def _test_binary_op(precision, device_id, op_func, left_operand, right_operand,
                    expected_forward, expected_backward_all, only_input_variables=False):

    left_value = AA(left_operand, dtype=PRECISION_TO_TYPE[precision])
    right_value = AA(right_operand, dtype=PRECISION_TO_TYPE[precision])

    a = I(shape=left_value.shape,
          data_type=sanitize_dtype_cntk(precision),
          needs_gradient=True,
          name='a')

    b = I(shape=right_value.shape,
          data_type=sanitize_dtype_cntk(precision),
          needs_gradient=True,
          name='b')

    if (type(op_func) == str):
        input_op_constant = eval('a %s right_operand' % op_func)
        constant_op_input = eval('left_operand %s b' % op_func)
        input_op_input = eval('a %s b' % op_func)
    else:
        input_op_constant = op_func(a, right_value)
        constant_op_input = op_func(left_value, b)
        input_op_input = op_func(a, b)

    # create batch by wrapping the data point into a sequence of length one and
    # putting it into a batch of one sample
    left_value.shape = (1, 1) + left_value.shape
    right_value.shape = (1, 1) + right_value.shape

    forward_input = {a: left_value, b: right_value}
    expected_backward = {a: expected_backward_all[
        'left_arg'], b: expected_backward_all['right_arg'], }
    unittest_helper(input_op_input,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)

    if not only_input_variables:
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
    forward, backward = cntk_eval(root_node, precision, cntk_device(device_id),
                                  forward_input, backward_pass)

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
        batch (list): list of samples. If `dynamic_axis` is given, samples are sequences
         of tensors. Otherwise, they are simple tensors.
        dynamic_axis (str or :func:`cntk.ops.dynamic_axis` instance): the dynamic axis

    Returns:
        (indices, values, shape)
    '''

    batch_indices = []
    batch_values = []
    tensor_shape = []

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
