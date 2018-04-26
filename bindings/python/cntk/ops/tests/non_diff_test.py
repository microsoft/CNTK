# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for operations that are not differentiable.
"""

from __future__ import division
import numpy as np
import cntk as C
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, precision, PRECISION_TO_TYPE
from cntk.internal import sanitize_dtype_cntk

TENSORS = [
    ([12.3, -12.3]),
    ([10.2, -10.2]),
    ([0.5, -0.5]),
    ([0.01, -0.01]),
    ([0.499, -0.499]),
    ([5.0, -5.0]),
    ([0.0]),
    ([[2.1, 9.9], [4.7, 5.3]])
]


@pytest.mark.parametrize("operand", TENSORS)
def test_op_floor(operand, device_id, precision):
    operand = AA(operand)
    expected = np.floor(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import floor
    _test_unary_op(precision, device_id, floor, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_ceil(operand, device_id, precision):
    operand = AA(operand)
    expected = np.ceil(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import ceil
    _test_unary_op(precision, device_id, ceil, operand,
                   expected_forward, expected_backward)

# Manually setting the expectation since CNTK's round behaves differently than
# NumPy's round (see operator's docstring).
ROUND_TENSORS = [
    ([0.2, 1.3, 4.0, 5.5, 0.0],
     [0.0, 1.0, 4.0, 6.0, 0.0]),

    ([[0.6, 3.3], [1.9, 5.6]],
     [[1.0, 3.0], [2.0, 6.0]]),

    ([-5.5, -4.2, -3., -0.7, 0],
     [-5.0, -4.0, -3., -1.0, 0]),

    ([[-0.6, -4.3], [1.9, -3.2]],
     [[-1.0, -4.0], [2.0, -3.0]]),

    # CNTK is always rounding up values starting at x.5, while numpy rounds
    # to the nearest even value for half-integers
    # Refer here: https://en.wikipedia.org/wiki/Rounding#Tie-breaking
    # This test shows such values are not equal comparing numpy and CNTK
    ([0.5, 1.5, 2.5, 3.5],
     # NumPy would round to
     #        [0.0, 2.0, 2.0, 4.0]))
     # while CNTK rounds to
     [1.0, 2.0, 3.0, 4.0])
]


@pytest.mark.parametrize("operand,expected", ROUND_TENSORS)
def test_op_round(operand, expected, device_id, precision):
    operand, expected = AA(operand), AA(expected)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import round
    _test_unary_op(precision, device_id, round, operand,
                   expected_forward, expected_backward)

def test_input_variable():
    from .. import sequence
    i = C.input_variable(shape=(2,3), name='i')
    assert i.shape == (2,3)
    assert i.name == 'i'
    assert len(i.dynamic_axes)==1

    sequence_i = sequence.input_variable(shape=(3,2), name='sequence_i')
    assert sequence_i.shape == (3,2)
    assert sequence_i.name == 'sequence_i'
    assert len(sequence_i.dynamic_axes)==2


@pytest.mark.parametrize("operand", TENSORS)
def test_zeros_like(operand, device_id, precision):
    operand = AA(operand)
    expected = np.zeros_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import zeros_like
    _test_unary_op(precision, device_id, zeros_like, operand,
                   expected_forward, expected_backward)

def test_zeros_like_empty_shape(device_id, precision):
    operand = np.array(3)
    expected = np.zeros_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import zeros_like
    _test_unary_op(precision, device_id, zeros_like, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_ones_like(operand, device_id, precision):
    operand = AA(operand)
    expected = np.ones_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import ones_like
    _test_unary_op(precision, device_id, ones_like, operand,
                   expected_forward, expected_backward)

Matrices = [
    ([[2.1, 4.7], [2.1, 2.1]], True),
    ([[2.1, 2., 2.], [4.7, 3, 5], [5.1, 2, 5]], True),
    ([[2.1], [4.7], [5.1], [5.8]], True),
    ([[2.1, 4.7], [2.1, 2.1]], False),
    ([[2.1, 2., 2.], [4.7, 3, 5], [5.1, 2, 5]], False),
    ([[2.1], [4.7], [5.1], [5.8]], False),
]

NO_BACKPROP_TEST_OPERANDS = [
    # (input_data, )
    ([[1]], ),
    ([[1, 2], [4, 5]], ),
    ([[1, 2], [4, 5]], ),
    ([[1, 2], [4, 5]], ),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ),
]


@pytest.mark.parametrize("input_data", NO_BACKPROP_TEST_OPERANDS)
def test_constant_like_no_backprop(input_data, precision):
    from .. import eye_like, ones_like, zeros_like
    no_backprop_ops = [ones_like, zeros_like]
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    x = C.input_variable(shape=data.shape,
              dtype=sanitize_dtype_cntk(dt),
              needs_gradient=True,
              name='a')
    w = C.parameter(x.shape, init=np.ones(x.shape).astype(dt) * 3.0)
    # create batch
    data.shape = (1,) + data.shape
    expected_x_backward = np.zeros_like(data)
    expected_w_backward = np.zeros_like(w)

    # numpy argmax doesn't support keepdims
    for op in no_backprop_ops:
        #test direct input: no gradients pass through to inputs
        op_func = op(x)
        grad = op_func.grad({x: data}, [x])
        np.testing.assert_almost_equal(grad, expected_x_backward)

        #test inputs through sub-expressions: no gradients pass through to inputs (e.g. x, w) of the subexpressoin (e.g. x * w here)
        op_func = op(x * w)
        grad = op_func.grad({x: data}, [w, x])
        np.testing.assert_almost_equal(grad[x], expected_x_backward)
        np.testing.assert_almost_equal(grad[w], expected_w_backward)

        #testing inputs through shared sub-expressions: no gradients pass through reduce arg ops to inputs (e.g. x, w) of the subexpressoin
        # (e.g. x * w here), therefore the gradients will depend on how the shared expressions participate in other experssions:
        shared_exp = x * w
        op_func = op(shared_exp) + x + w + shared_exp
        ref_op_func = x + w + shared_exp
        grad = op_func.grad({x: data}, [w, x])
        ref_grad = ref_op_func.grad({x: data}, [w, x])
        np.testing.assert_almost_equal(grad[x], ref_grad[x])
        np.testing.assert_almost_equal(grad[w], ref_grad[w])
