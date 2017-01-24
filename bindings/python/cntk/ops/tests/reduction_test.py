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

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_all(input_data, axis, device_id, precision):
    # FIXME: we'd like to do dt = PRECISION_TO_TYPE[precision]
    # however there seems to be an issue with actual_forward below
    # that gets computed correctly but by the time np.allclose executes
    # it contains garbage values. The problem goes away if one uses 
    # actual_forward  = np.copy(input_op.eval(binding))
    dt = np.float32
    data = AA(input_data, dtype=dt)
    a = I(shape=data.shape,
          dtype=sanitize_dtype_cntk(dt),
          needs_gradient=True,
          name='a')
    # create batch
    value = AA([[data,data+0.1],[data+0.2,data+0.3]], dtype=dt)
    from .. import reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_log_sum
    from cntk import Axis
    def max_bwd(x):
        out = np.zeros_like(x)
        out[np.unravel_index(np.argmax(x),x.shape)] = 1
        return out
    def min_bwd(x):
        out = np.zeros_like(x)
        out[np.unravel_index(np.argmin(x),x.shape)] = 1
        return out
    ops = [ (reduce_sum,     lambda x:AA(np.sum(x)),                 lambda x:np.ones_like(x)),
            (reduce_max,     lambda x:AA(np.max(x)),                 lambda x:max_bwd(x)),
            (reduce_min,     lambda x:AA(np.min(x)),                 lambda x:min_bwd(x)),
            (reduce_mean,    lambda x:AA(np.mean(x)),                lambda x:np.ones_like(x) / x.size),
            (reduce_log_sum, lambda x:AA(np.log(np.sum(np.exp(x)))), lambda x:np.exp(value)/np.sum(np.exp(x)))]
    
    for op,fwd,bwd in ops:
        input_op = op(a, axis=Axis.all_axes())
        expected_forward = fwd(value)
        expected_backward = bwd(value)
        binding = {a: value}
        actual_backward = input_op.grad(binding)[0]
        actual_forward  = np.copy(input_op.eval(binding))
        assert np.allclose(actual_forward, expected_forward)
        assert np.allclose(actual_backward, expected_backward)

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_mean_all_constant(input_data, axis, device_id, precision):
    # dt = PRECISION_TO_TYPE[precision]
    # FIXME: we'd like to do dt = PRECISION_TO_TYPE[precision]
    # however there seems to be an issue with actual_forward below
    # that gets computed correctly but by the time np.allclose executes
    # it contains garbage values. The problem goes away if one uses 
    # actual_forward  = np.copy(input_op.eval())
    dt = np.float32
    value = AA(input_data, dtype=dt)
    from .. import reduce_mean
    from cntk import Axis, Constant
    a = Constant(value, name='a')
    input_op = reduce_mean(a, axis=Axis.all_axes())
    expected_forward = AA(np.mean(value))
    actual_forward  = input_op.eval()
    assert np.allclose(actual_forward, expected_forward)