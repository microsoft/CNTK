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
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE

EPS_IN_LOG = 1e-37        # 1e-37 is the highest guaranteed precision
BACKWARD_RESULST_FOR_LOG_EPS = 9.08782e+36 # the backward result returned by CNTK log() for epsilon
LOG_OF_EPS_IN_LOG =  -85.1 # log(EPS_IN_LOG)

CLIP_TUPLES = [
    ([1.0], [2.0], [1.5]), # value shouldn't be clipped; gradient is [1.0]
    ([1.0], [2.0], [0.5]), # value should be clipped to 1.0; gradient is [0.0]
    ([1.0], [2.0], [2.5]), # value should be clipped to 2.0; gradient is [0.0]
    
    # should clip to [1.5, 2.0, 1.0]; gradient is [[1.0, 0.0, 0.0]]
    ([1.0], [2.0], [[1.5, 2.1, 0.9]]),

    # should clip to [[1.0, 2.0], [1.0, 2.0], [1.5, 2.0]];
    # gradient is [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ([1.0], [2.0], [[0.0, 3.0], [1.0, 2.0], [1.5, 2.5]]),
     
    # test what happens if a user puts a higher "min" value than their "max" value
    # should clip to [[5.0, 5.0, 5.0, 5.0, 5.0]] because min is evaluated first
    # gradient should be all zeros: [[0.0, 0.0, 0.0, 0.0, 0.0]]
    ([5.0], [0.5], [[1.5, 2.1, 0.9, -1.0, -2.0]]),
     
    # test a more complicated broadcasting scenario
    ([[1.5, 2.0], [2.5, 3.0]], [[-2.0, 2.5], [2.5, 3.5]], [[-1.0, 2.0], [3.0, 4.0]]),
    ]
@pytest.mark.parametrize("min_value, max_value, x", CLIP_TUPLES)
def test_op_clip(min_value, max_value, x, device_id, precision):    
    from .. import clip

    expected_forward = [np.clip(AA([x], dtype=PRECISION_TO_TYPE[precision]), AA(min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))]

    expected_backward = {
            'arg': [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]
            }

    _test_unary_op(precision, device_id, clip, x,
            expected_forward, expected_backward, 
            {'min_value': min_value, 'max_value': max_value})
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
def test_op_abs(operand, device_id, precision):
    t = np.abs(AA(operand, dtype=PRECISION_TO_TYPE[precision]))

    expected_forward = [AA([t])]

    # For 0 NumPy gives a gradient non, while CNTK gives 0
    backward = operand / np.abs(operand)
    backward[np.isnan(backward)] = 0
    expected_backward = {
            'arg': [[backward]]
            }

    from .. import abs
    _test_unary_op(precision, device_id, abs, operand,
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
@pytest.mark.parametrize("shape", [(3,9), (10,20,30)])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2, 0.5, 0.8])
def test_op_dropout(shape, dropout_rate, device_id, precision):
    from cntk import dropout
    from cntk.utils import eval, sanitize_dtype_cntk, cntk_device

    count = 10
    resulted_non_zeros = 0

    # As the dropout node is stochastic, we run it a couple times and aggregate
    # over the results to get more stable tests.
    for i in range(count):
        value = np.ones(shape=shape, dtype=PRECISION_TO_TYPE[precision])

        a = I(shape=value.shape,
                data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                needs_gradient=True,
                name='a')

        dropout_node = dropout(a, dropout_rate=dropout_rate)

        value.shape = (1,1) + value.shape    
        forward_input = {a:value}    

        forward, backward = eval(dropout_node, 
                precision, 
                cntk_device(device_id), 
                forward_input, 
                backward_pass=True)

        resulted_non_zeros += np.count_nonzero(forward[dropout_node.output()])

    resulted_non_zeros /= count
    num_elements = np.multiply.reduce(shape)
    expected_non_zeros = num_elements * (1-dropout_rate) 
    max_off = 0.2*num_elements

    assert(abs(resulted_non_zeros-expected_non_zeros) <
            max_off)
    
@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 100])
def test_op_dropout_bad_input(dropout_rate):
    from cntk import dropout
    from cntk.utils import eval, sanitize_dtype_cntk, cntk_device

    a = I(shape=(1,2), data_type='float', needs_gradient=True, name='a')

    with pytest.raises(ValueError):
        dropout_node = dropout(a, dropout_rate=dropout_rate)

