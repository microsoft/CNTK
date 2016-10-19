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
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

TARGET_OUT_PAIRS = [
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0.5, 0.5]], [[1., 2., 3., 4.]]),
    ([[0., 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
]

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_cross_entropy_with_soft_max(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    s_max = exp_x / np.sum(exp_x)

    t = AA(target_vector, dtype=dt)

    expected_forward = np.asarray(-np.sum(t * np.log(s_max, dtype=dt),
        dtype=dt))
    expected_forward.shape = (1,1,1,1)+expected_forward.shape

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

TARGET_OUT_PAIRS_EP = [
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
]

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS_EP)
def test_op_classification_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    different_position = np.argmax(t) != np.argmax(o)

    expected_forward = [[AA([[int(different_position)]], dtype=dt)]]

    zero_backward = np.zeros_like([[t]], dtype=dt)
    left_backward = np.copy(zero_backward)
    
    zero_backward[..., np.argmax(o)] = -1.
    right_backward = zero_backward

    expected_backward = {
        'left_arg':  left_backward,
        'right_arg': right_backward
    }
    
    from .. import classification_error
    _test_binary_op(precision, device_id, classification_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward)
