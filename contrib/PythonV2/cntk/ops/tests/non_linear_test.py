# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operations. Each operation is tested for 
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type

EPS_IN_LOG = 1e-37        # 1e-37 is the highest guaranteed precision
BACKWARD_RESULST_FOR_LOG_EPS = 9.08782e+36 # the backward result returned by CNTK log() for epsilon
LOG_OF_EPS_IN_LOG =  -85.1 # log(EPS_IN_LOG)

TENSORS = [
    ([[0, -0.1]]),
    ([[-100, -10], [-1, -0.1], [-0.01, -0.001],
      [0.001, 0.01], [0.1, 1], [10, 100]]),
]

@pytest.mark.parametrize("operand", TENSORS)
def test_op_sigmoid(operand, device_id, precision):   
    s = 1.0 / (1.0 + np.exp(-AA(operand, dtype=PRECISION_TO_TYPE[precision])))
    expected_forward = [AA([s])]

    expected_backward = {
            'arg': [[s * (1 - s)]],            
            }

    from .. import sigmoid    
    _test_unary_op(precision, device_id, sigmoid, operand, 
        expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_exp(operand, device_id, precision):
    e = np.exp(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = [AA([e])]

    expected_backward = {
            'arg': expected_forward,            
            }

    from .. import exp
    _test_unary_op(precision, device_id, exp, operand, 
        expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_tanh(operand, device_id, precision):    
    t = np.tanh(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = [AA([t])]

    expected_backward = {
            'arg': [[1 - t**2]],
            }
    
    from .. import tanh
    _test_unary_op(precision, device_id, tanh, operand, 
        expected_forward, expected_backward)
