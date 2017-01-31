# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, cntk_device
from ...utils import sanitize_dtype_cntk, _ones_like, eval

TENSOR_PAIRS = [
    ([30.], [10.]),
    ([[10.]], [[30.]]),
    ([[1.5, 2.1]], [[10., 20.]]),
    ([[100., 200.], [300., 400.], [10., 20.]],
     [[10., 20.], [30., 40.], [1., 2.]]),

    # Adding two 3x2 inputs of sequence length 1
    ([[30., 40.], [1., 2.], [0.1, 0.2]], [[10, 20], [3, 4], [-0.5, -0.4]]),
]

# -- plus operation tests --
TENSOR_PAIRS_SCALAR = TENSOR_PAIRS + [(left, np.random.rand()) for left, right
                                      in TENSOR_PAIRS]


@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS_SCALAR)
def test_op_plus(left_operand, right_operand, device_id, precision):
    expected_forward = [AA([left_operand]) + AA([right_operand])]

    if np.isscalar(right_operand):
        expected_backward = {
            'left_arg':  [[[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]]],
            # gradients are accumulated
            'right_arg': [[AA([left_operand]).size]]
        }
    else:
        expected_backward = {
            'left_arg':  [[[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]]],
            'right_arg': [[[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in right_operand]]]
        }
    from .. import plus
    _test_binary_op(precision, device_id, plus,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '+',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

def test_op_plus_sequences(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]
    operand = [AA([[1., 2.], [3., 4.]], dtype=dt_precision), AA([[5., 6.]], dtype=dt_precision)]
    root_gradient = [AA([[1., 1.], [1., 1.]], dtype=dt_precision), AA([[1., 1.]], dtype=dt_precision)]

    expected_forward = [AA([[2., 4.], [6., 8.]], dtype=dt_precision), AA([[10., 12.]], dtype=dt_precision)]
    expected_backward = [AA([[2., 2.], [2., 2.]], dtype=dt_precision), AA([[2., 2.]], dtype=dt_precision)]

    from .. import plus, input_variable
    x = input_variable(shape=(2,), needs_gradient=True)
    z = x + x
    state, actual_forward = z.forward({x : operand}, [z.output], {z.output}, cntk_device(device_id))
    actual_backward = z.backward(state, {z.output : root_gradient}, [x])

    np.allclose(list(actual_forward.values())[0][0], expected_forward[0])
    np.allclose(list(actual_forward.values())[0][1], expected_forward[1])

    np.allclose(list(actual_backward.values())[0][0], expected_backward[0])
    np.allclose(list(actual_backward.values())[0][1], expected_backward[1])

def test_op_plus_gradient_accumulation(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    value = AA([[[1]]], dtype=dt_precision)

    from cntk import times_transpose, Axis
    a = I(shape=(1,), dtype=dt_precision,
          needs_gradient=True,
          name='a')

    input_op = a + a

    expected_forward = AA([[[2]]], dtype=dt_precision)
    expected_backward = { a : [[[2]]], a : [[[2]]] }

    forward_input = {a: value}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


SEQ_TENSOR_PAIRS = [
    # two inputs each having sequences of length 1 and 2
    ([[[30.]], [[40], [50]]],  # first batch with two sequences
     [[[3.]], [[4], [5]]]),  # second batch with two sequences

    #([[[30.,   0]], [[40,   1], [50,   2]]],  # first batch with two sequences
     #[[[3., -10]], [[4, -20], [5, -30]]]),  # second batch with two sequences
]


@pytest.mark.parametrize("left_batch, right_batch", SEQ_TENSOR_PAIRS)
def test_op_plus_var_sequences_input_input(left_batch, right_batch, device_id, precision):
    from .. import plus

    assert len(left_batch) == len(right_batch)
    expected_forward = [AA(left_batch[i]) + AA(right_batch[i])
                        for i in range(len(left_batch))]

    expected_backward = {
        'left': _ones_like(left_batch, PRECISION_TO_TYPE[precision]),
        'right': _ones_like(right_batch, PRECISION_TO_TYPE[precision])
    }

    left_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision])
                  for sample in left_batch]
    left_shape = left_value[0][0].shape
    right_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision])
                   for sample in right_batch]
    right_shape = right_value[0][0].shape

    a = I(shape=left_shape,
          dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    b = I(shape=right_shape,
          dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='b')

    input_op_input = plus(a, b)
    forward_input = {a: left_value, b: right_value}
    backward_input = {a: None, b: None}
    expected_backward = {
            a: expected_backward['left'], 
            b: expected_backward['right'], }
    unittest_helper(input_op_input,
                    forward_input, expected_forward,
                    expected_backward,
                    device_id, precision)

# -- minus operation tests --
# TODO: enable once the function is exposed


@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_minus(left_operand, right_operand, device_id, precision):
    expected_forward = [AA([left_operand], dtype=PRECISION_TO_TYPE[
                           precision]) - AA([right_operand], dtype=PRECISION_TO_TYPE[precision])]

    expected_backward = {
        'left_arg':  [[[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]]],
        'right_arg': [[[-1 * np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in right_operand]]]
    }
    from .. import minus
    _test_binary_op(precision, device_id, minus,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '-',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

# -- element times tests --


@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_times(left_operand, right_operand, device_id, precision):
    expected_forward = [AA([left_operand]) * AA([right_operand])]

    expected_backward = {
        'left_arg':  [[right_operand]],
        'right_arg': [[left_operand]]
    }

    from .. import element_times
    _test_binary_op(precision, device_id, element_times,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '*',
                    left_operand, right_operand,
                    expected_forward, expected_backward)


# -- element divide tests --
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_divide(left_operand, right_operand, device_id, precision):
    expected_forward = [AA([left_operand]) / AA([right_operand])]

    expected_backward = {
        'left_arg':  [[[np.ones_like(x) / x for x in right_operand]]],
        'right_arg': [[-AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) / AA(right_operand, dtype=PRECISION_TO_TYPE[precision])**2]]
    }

    from .. import element_divide
    _test_binary_op(precision, device_id, element_divide,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '/',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

NEGATE_TENSORS = [
    ([30.]),
    ([[30.]]),
    ([[1.5, 2.1]]),
    ([[100., 200.], [300., 400.], [10., 20.]]),
    ([[30, 40], [1, 2], [0.1, 0.2]])
]


@pytest.mark.parametrize("operand", NEGATE_TENSORS)
def test_op_negate(operand, device_id, precision):
    t = -1 * AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [AA([t])]

    expected_backward = {
        'arg': [[-1 * np.ones_like(operand, PRECISION_TO_TYPE[precision])]]
    }

    from cntk import negate

    _test_unary_op(precision, device_id, negate, operand,
                   expected_forward, expected_backward)

    _test_unary_op(precision, device_id, '-', operand,
                   expected_forward, expected_backward)

# transpose_times currently only supports right operands of rank 1 or 2
TRANSPOSE_TIMES_PAIRS = [
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10.], [20.]]),
    ([[100., 200.]], [[-10.], [20.]]),
    ([[100., 200.], [300., 400.]], [[10.], [20.]]),
    ([[100., 200.], [-300., 400.]], [[10., 20.], [20., 30.]]),
    (np.reshape(np.arange(24), (4, 3, 2)),
     np.array([[1, 3], [2, 4]])),
]

# adding a rank 3 operand for times operation
TIMES_PAIRS = TRANSPOSE_TIMES_PAIRS + \
    list((np.reshape(np.arange(8), (2, 2, 2)), np.reshape(np.arange(8), (2, 2, 2))))


@pytest.mark.parametrize("left_operand, right_operand", TIMES_PAIRS)
def test_op_times(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)

    expected_forward = [[np.tensordot(a, b, axes=len(b.shape) - 1)]]

    left_backward = np.zeros_like(a)
    left_backward[...] = b.sum(axis=-1)

    right_backward = np.zeros_like(b)
    transpose_axes = list(np.roll(np.arange(len(b.shape)), -1))
    sum_axes = tuple(np.arange(0, len(a.shape) - len(b.shape) + 1))
    right_backward[...] = np.transpose(
        AA([a.sum(axis=sum_axes)]), axes=transpose_axes)

    expected_backward = {
        'left_arg':  [[left_backward]],
        'right_arg': [[right_backward]]
    }

    from cntk import times

    _test_binary_op(precision, device_id, times,
                    left_operand, right_operand, expected_forward, expected_backward)


@pytest.mark.parametrize("left_operand, right_operand", TRANSPOSE_TIMES_PAIRS)
def test_op_transpose_times(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    # tranpose right_operand to make product possible
    right_operand = np.transpose(right_operand).tolist()

    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)

    expected_forward = [[np.dot(a, np.transpose(b))]]

    left_backward = np.zeros_like(a)
    left_backward[...] = b.sum(axis=tuple(range(len(b.shape) - 1)))

    right_backward = np.zeros_like(b)
    right_backward[...] = a.sum(axis=tuple(range(len(a.shape) - 1)))

    expected_backward = {
        'left_arg':  [[left_backward]],
        'right_arg': [[right_backward]]
    }

    from cntk import times_transpose

    _test_binary_op(precision, device_id, times_transpose,
                    left_operand, right_operand, expected_forward, expected_backward)

