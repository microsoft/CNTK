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
from .ops_test_utils import _test_binary_op, AA, I, precision, PRECISION_TO_TYPE,\
        unittest_helper

from cntk import edit_distance_error, input_variable

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
    expected_forward.shape = (1,1,1,1) + expected_forward.shape

    s = np.sum(t, dtype=dt)
    backward = np.subtract(s_max * s, t)
    backward.shape = (1,1) + backward.shape

    expected_backward = {
        'left_arg':  backward,
        'right_arg': [[-1*o]]
    }

    from .. import cross_entropy_with_softmax
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

    expected_forward = [[np.reshape(AA(expected_forward, dtype=dt), (x.shape[0], 1))]]
    expected_backward_left = AA(expected_backward_left, dtype=dt)

    expected_backward = {
        'left_arg':  [[expected_backward_left]],
        'right_arg': [[expected_backward_right]]
    }

    from .. import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward, op_param_dict={'axis': axis})

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_squared_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = AA([[np.sum((t - o)**2)]])

    backward = 2 * np.subtract(o, t)
    expected_backward = {
        'left_arg':  [[backward]],
        'right_arg': [[-1*backward]]
    }

    from .. import squared_error
    _test_binary_op(precision, device_id, squared_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_CLASSIFICATION = [
    # (target_vector, output_vector)
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
]

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS_CLASSIFICATION)
def test_op_classification_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    different_position = np.argmax(t) != np.argmax(o)

    expected_forward = [[AA([[int(different_position)]], dtype=dt)]]

    zero_backward = np.zeros_like(t, dtype=dt)
    left_backward = np.copy(zero_backward)

    zero_backward[..., np.argmax(o)] = -1.
    right_backward = zero_backward

    expected_backward = {
        'left_arg':  [[left_backward]],
        'right_arg': [[right_backward]]
    }

    from .. import classification_error
    _test_binary_op(precision, device_id, classification_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_CLASSIFICATION_WITH_AXIS = [
    # (target_vector, output_vector, axis)
    ([[0., 1., 0., 0.],
      [0., 1., 0., 0.]],
     [[1., 2., 3., 4.],
      [1., 5., 3., 4.]], 1),
    ([[0., 1., 0., 0.],
      [0., 0., 1., 0.],
      [0., 1., 0., 0.]],
     [[1., 2., 3., 4.],
      [6., 2., 7., 4.],
      [1., 5., 3., 4.]], 1),
    ([[0., 0., 0.5, 0.5],
      [0., 0., 1., 0.],
      [0., 1., 0., 0.]],
     [[1., 2., 3., 4.],
      [6., 2., 7., 4.],
      [1., 5., 3., 4.]], 1),
]

@pytest.mark.parametrize("target_vector, output_vector, axis", TARGET_OUT_PAIRS_CLASSIFICATION_WITH_AXIS)
def test_op_classification_error_with_axis(output_vector, target_vector, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    x = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    forward = []
    expected_backward_left = []
    expected_backward_right = []

    for sample, target in zip(x, t):
        different_position = np.argmax(target) != np.argmax(sample)
        forward.append([int(different_position)])

        zero_backward = np.zeros_like(target, dtype=dt)

        expected_backward_left.append(zero_backward)
        expected_backward_right.append(zero_backward)

    forward = np.mean(forward)

    expected_forward = AA([[forward]], dtype=dt)
    expected_backward_left = AA([[expected_backward_left]], dtype=dt)
    expected_backward_right = AA([[expected_backward_right]], dtype=dt)

    expected_backward = {
        'left_arg':  expected_backward_left,
        'right_arg': expected_backward_right
    }

    from .. import classification_error
    _test_binary_op(precision, device_id, classification_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward, op_param_dict={'axis':axis})

LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS = [
    # (grad, value, output, gain)
    ([[[-0.2121461]],  [[ 0.2121461]]],  58.038055419921875, [1, 2], [7, 1]),
    ([[[-0.14861868]], [[ 0.14861868]]], 40.65847396850586,  [3, 4], [3, 1])
]

@pytest.mark.parametrize("grad, value, output, gain", LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS)
def test_lambda_rank(grad, value, output, gain, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    score = AA(output, dtype=dt).reshape(-1,1,1)
    gain  = AA(gain, dtype=dt).reshape(-1,1,1)
    group = np.ones_like(score).reshape(-1,1,1)

    expected_value = AA(value, dtype=dt)
    expected_grad  = AA(grad, dtype=dt)

    from .. import lambda_rank

    g = I((1,))
    s = I((1,), needs_gradient=True)
    n = I((1,))
    f = lambda_rank(s, n, g)

    actual_grad  = f.grad({s:score, n:gain, g:group}, [s])[0]
    actual_value = f.eval({s:score, n:gain, g:group})

    assert np.allclose(actual_value, expected_value)
    assert np.allclose(actual_grad,  expected_grad)

NDCG_VALUES_AND_INPUTS = [
    # (value, output, gain)
    (200, [2, 1],    [7, 1]),
    (300, [4, 2, 1], [3, 2, 1])
]

@pytest.mark.parametrize("value, output, gain", NDCG_VALUES_AND_INPUTS)
def test_ndcg(value, output, gain, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    score = AA(output, dtype=dt).reshape(-1,1,1)
    gain  = AA(gain, dtype=dt).reshape(-1,1,1)
    group = np.ones_like(score).reshape(-1,1,1)

    expected_value = AA(value, dtype=dt)

    from .. import ndcg_at_1

    g = I((1,))
    s = I((1,))
    n = I((1,))
    f = ndcg_at_1(s, n, g)

    actual_value = f.eval({s:score, n:gain, g:group})

    assert np.allclose(actual_value, expected_value)

EDIT_DISTANCE_ERROR_TEST_CASES = [
        # drawing 1 sample
    ([[1, 2]], [[1, 2]], 0, 0, 0, False, [], 0.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 0, 0, False, [], 1.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 1, 0, 0, False, [], 2.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 1, 1, False, [], 1.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 1, 1, True, [1], 2.0),
]

@pytest.mark.parametrize("left_input, right_input, subPen, delPen, insPen, squashInputs, samplesToIgnore, result", EDIT_DISTANCE_ERROR_TEST_CASES)
def test_edit_distance_error(left_input, right_input, subPen, delPen, insPen, squashInputs, samplesToIgnore, result, device_id, precision):
    i1 = input_variable(shape=(2,))
    i2 = input_variable(shape=(2,))
    arguments = {i1 : left_input, i2 : right_input}
    a = edit_distance_error(i1, i2, subPen, delPen, insPen, squashInputs, samplesToIgnore)
    assert np.allclose(result, a.eval(arguments))