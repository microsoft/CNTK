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
    #([[0., 0., 0.5, 0.5]], [[1., 2., 3., 4.]]),
    #([[0., 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
]

# TODO: Enable tests when 0d arrays are correctly handled for backward
# propagation e. g. array(29.0)


@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def _test_op_cross_entropy_with_soft_max(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    
    def numpy_softmax(x):
        x = AA(x, dtype=PRECISION_TO_TYPE[precision])
        ox = x - x.max()  # subtract max to avoid overflow

        expX = np.exp(ox)
        return expX / np.sum(expX)
    
    def numpy_op(label, softmax):
        return -np.sum(label * np.log(softmax, dtype=PRECISION_TO_TYPE[precision]), dtype=PRECISION_TO_TYPE[precision])

    def numpy_grad(softmax, target):
        s = np.sum(target, dtype=dt) #This should be 1.0
        return np.subtract(softmax * s , target)


    expected_forward = [[[[numpy_op(AA(target_vector, dtype=PRECISION_TO_TYPE[precision]), 
                           numpy_softmax(output_vector))]]]]

    backward = [[numpy_grad(numpy_softmax(output_vector), 
                AA(target_vector, dtype=PRECISION_TO_TYPE[precision]))]]

    expected_backward = {
        'left_arg':  backward,
        'right_arg': backward
    }

    from .. import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward, True)


@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_squared_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = [[np.sum((t - o)**2)]]

    expected_backward = {
        'left_arg':  [[-2 * np.subtract(t, o)]],
        'right_arg': [[-2 * np.subtract(o, t)]]
    }

    from .. import squared_error
    _test_binary_op(precision, device_id, squared_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward, True)

TARGET_OUT_PAIRS_EP = [
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
]

# -- ErrorPrediction with softmax operation tests --

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS_EP)
def test_op_classification_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = [[int(np.argmax(t) != np.argmax(o))]]
    
    from .. import classification_error
    op = classification_error(o, t)
    
    unittest_helper(op, {}, expected_forward, {},
            device_id=device_id, precision=precision)
