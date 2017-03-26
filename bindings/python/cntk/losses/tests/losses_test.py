# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evaluation operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from cntk.ops.tests.ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

from cntk import input

TARGET_OUT_PAIRS = [
    # (target_vector, output_vector)
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0.5, 0.5]], [[1., 2., 3., 4.]]),
    ([[0., 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
]

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_cross_entropy_with_soft_max(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    ox = o - o.max()  # subtract max to avoid overflow
    exp_x = np.exp(ox)
    s_max = exp_x / np.sum(exp_x) # softmax function

    expected_forward = np.asarray(-np.sum(t * np.log(s_max, dtype=dt), dtype=dt))
    expected_forward.shape = (1,1,1) + expected_forward.shape

    s = np.sum(t, dtype=dt)
    backward = np.subtract(s_max * s, t)
    backward.shape = (1,) + backward.shape

    expected_backward = {
        'left_arg':  backward,
        'right_arg': [-1*o]
    }

    from cntk.losses import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_WITH_AXIS = [
    # (target_vector, output_vector, axis)
    ([[0., 0., 0., 1]],
     [[1., 2., 3., 4.]], -1),
    ([[0., 0., 0.5, 0.5]],
     [[1., 2., 3., 4.]], 1),
    ([[0., 0.4, 0.3, 0.3]],
     [[2., 1., 1., 4.]], 1),
    ([[0., 0., 0., 1],
      [0., 0., 1., 0.]],
     [[1., 2., 3., 4.],
      [1., 2., 3., 5.]], 1),
    ([[0., 0., 0., 1],
      [0., 1., 0., 0.]],
     [[1., 2., 3., 4.],
      [1., 7., 3., 5.]], 1)
]

@pytest.mark.parametrize("target_vector, output_vector, axis", TARGET_OUT_PAIRS_WITH_AXIS)
def test_op_cross_entropy_with_soft_max_and_axis(output_vector, target_vector, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    x = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = []
    expected_backward_left = []
    expected_backward_right = []

    for sample, target in zip(x, t):
        ox = sample - sample.max()  # subtract max to avoid overflow
        exp_x = np.exp(ox)
        s_max = exp_x / np.sum(exp_x) # softmax function
        forward = np.asarray(-np.sum(target * np.log(s_max, dtype=dt), dtype=dt))
        expected_forward.append(forward.tolist())

        s = np.sum(target, dtype=dt)
        backward = np.subtract(s_max * s, target)

        expected_backward_left.append(backward.tolist())
        expected_backward_right.append(-1*sample)

    expected_forward = [np.reshape(AA(expected_forward, dtype=dt), (x.shape[0], 1))]
    expected_backward_left = AA(expected_backward_left, dtype=dt)

    expected_backward = {
        'left_arg':  [expected_backward_left],
        'right_arg': [expected_backward_right]
    }

    from cntk.losses import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward, op_param_dict={'axis': axis})

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_squared_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = AA([np.sum((t - o)**2)])

    backward = 2 * np.subtract(o, t)
    expected_backward = {
        'left_arg':  [backward],
        'right_arg': [-1*backward]
    }

    from cntk.losses import squared_error
    _test_binary_op(precision, device_id, squared_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_CLASSIFICATION = [
    # (target_vector, output_vector)
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
]

LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS = [
    # (grad, value, output, gain)
    ([[-0.2121461],  [ 0.2121461]],  58.038055419921875, [1, 2], [7, 1]),
    ([[-0.14861868], [ 0.14861868]], 40.65847396850586,  [3, 4], [3, 1])
]

@pytest.mark.parametrize("grad, value, output, gain", LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS)
def test_lambda_rank(grad, value, output, gain, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    score = AA(output, dtype=dt).reshape(-1,1,1)
    gain  = AA(gain, dtype=dt).reshape(-1,1,1)
    group = np.ones_like(score).reshape(-1,1,1)

    expected_value = AA(value, dtype=dt)
    expected_grad  = AA(grad, dtype=dt)

    from cntk.losses import lambda_rank

    g = input((1,))
    s = input((1,), needs_gradient=True)
    n = input((1,))
    f = lambda_rank(s, n, g)

    actual_grad, actual_value = f.grad({s:score, n:gain, g:group}, [s], [f.output])

    assert np.allclose(actual_value, expected_value)
    assert np.allclose(actual_grad,  expected_grad)
