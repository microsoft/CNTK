# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for reduction operations, tested for the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, AA, I, precision, PRECISION_TO_TYPE, constant
from ...utils import sanitize_dtype_cntk

REDUCE_TEST_OPERANDS = [
    #(input_data,  axis)
    ([[1]], 0),
    ([[1,2],[4,5]], 0),
    ([[1,2],[4,5]], 1),
    ([[1,2],[4,5]], -1),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], -2),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], 2),
]

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_sum(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    expected_forward = [[np.sum(data, axis=(axis), keepdims=True)]]

    backward = np.ones_like(data)

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_sum
    _test_unary_op(precision, device_id, reduce_sum, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_max(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    expected_forward = [[np.amax(data, axis=(axis), keepdims=True)]]

    forward_array = np.asarray(expected_forward, dtype=dt)
    max_elements = forward_array.reshape(forward_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(data)
    for element in max_elements:
        backward += np.asarray(data == element)

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_max
    _test_unary_op(precision, device_id, reduce_max, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_min(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    expected_forward = [[np.amin(data, axis=(axis), keepdims=True)]]

    forward_array = np.asarray(expected_forward, dtype=dt)
    max_elements = forward_array.reshape(forward_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(data)
    for element in max_elements:
        backward += np.asarray(data == element)

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_min
    _test_unary_op(precision, device_id, reduce_min, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_mean(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    expected_forward = [[np.mean(data, axis=(axis), keepdims=True)]]

    backward = np.ones_like(data) / data.shape[axis]

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_mean
    _test_unary_op(precision, device_id, reduce_mean, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_mean(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    expected_forward = [[np.mean(data, axis=(axis), keepdims=True)]]

    backward = np.ones_like(data) / data.shape[axis]

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_mean
    _test_unary_op(precision, device_id, reduce_mean, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_log_sum(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    data_exp = np.exp(data)
    sum_exp = np.sum(data_exp, axis=(axis), keepdims=True)
    expected_forward = [[np.log(sum_exp)]]

    backward = data_exp / sum_exp

    expected_backward = {
        'arg': [[backward]]
    }

    from .. import reduce_log_sum
    _test_unary_op(precision, device_id, reduce_log_sum, input_data,
                   expected_forward, expected_backward, {'axis': axis})
