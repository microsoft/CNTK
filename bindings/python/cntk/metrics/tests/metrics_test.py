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
import cntk as C
import pytest
from cntk.ops.tests.ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

from cntk import edit_distance_error, dropout


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

    expected_forward = [AA([[int(different_position)]], dtype=dt)]

    zero_backward = np.zeros_like(t, dtype=dt)
    left_backward = np.copy(zero_backward)

    zero_backward[..., np.argmax(o)] = -1.
    right_backward = zero_backward

    expected_backward = {
        'left_arg':  [left_backward],
        'right_arg': [right_backward]
    }

    from cntk.metrics import classification_error
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

    expected_forward = AA([forward], dtype=dt)
    expected_backward_left = AA([expected_backward_left], dtype=dt)
    expected_backward_right = AA([expected_backward_right], dtype=dt)

    expected_backward = {
        'left_arg':  expected_backward_left,
        'right_arg': expected_backward_right
    }

    from cntk.metrics import classification_error
    _test_binary_op(precision, device_id, classification_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward, op_param_dict={'axis':axis})

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

    from cntk.metrics import ndcg_at_1

    g = C.input_variable((1,))
    s = C.input_variable((1,))
    n = C.input_variable((1,))
    f = ndcg_at_1(s, n, g)

    actual_value = f.eval({s:score, n:gain, g:group})

    assert np.allclose(actual_value, expected_value)

EDIT_DISTANCE_ERROR_TEST_CASES = [
        # drawing 1 sample
    ([[1, 2]], [[1, 2]], 0, 0, 0, False, [], 0.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 0, 0, False, [], 1.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 1, 0, 0, False, [], 2.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 1, 1, False, [], 1.0),
    ([[1, 3], [2, 0]], [[2, 0], [2, 0]], 0, 1, 1, True, [1], 1.0),
]

@pytest.mark.parametrize("left_input, right_input, subPen, delPen, insPen, squashInputs, tokensToIgnore, result", EDIT_DISTANCE_ERROR_TEST_CASES)
def test_edit_distance_error(left_input, right_input, subPen, delPen, insPen, squashInputs, tokensToIgnore, result, device_id, precision):
    i1 = C.input_variable(shape=(2,))
    i2 = C.input_variable(shape=(2,))
    arguments = {i1 : left_input, i2 : right_input}
    a = edit_distance_error(i1, i2, subPen, delPen, insPen, squashInputs, tokensToIgnore)
    assert np.allclose(result, a.eval(arguments))
