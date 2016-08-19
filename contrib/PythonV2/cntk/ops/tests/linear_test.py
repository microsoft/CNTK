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
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type
from ...utils import sanitize_dtype_cntk, ones_like, eval, precision_numpy, cntk_device

TENSOR_PAIRS = [
    ([30.], [10.]),
    ([[10.]], [[30.]]),
    ([[1.5, 2.1]], [[10., 20.]]),
    #([[100., 200.], [300., 400.], [10., 20.]],
    #  [[10., 20.], [30., 40.], [1., 2.]]),    
    
    # Adding two 3x2 inputs of sequence length 1
    ([[30.,40.], [1.,2.], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]),
]

# -- plus operation tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_plus(left_operand, right_operand, device_id, precision):        
    expected_forward = [AA([left_operand]) + AA([right_operand])]

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

SEQ_TENSOR_PAIRS = [
    # two inputs each having sequences of length 1 and 2
    ([[[30.]], [[40], [50]]],  # first batch with two sequences
     [[[ 3.]], [[ 4], [ 5]]]), # second batch with two sequences

    ([[[30.,   0]], [[40,   1], [50,   2]]],  # first batch with two sequences
     [[[ 3., -10]], [[ 4, -20], [ 5, -30]]]), # second batch with two sequences
]
@pytest.mark.parametrize("left_batch, right_batch", SEQ_TENSOR_PAIRS)
def test_op_plus_var_sequences_input_input(left_batch, right_batch, device_id, precision):        
    assert len(left_batch) == len(right_batch)
    expected_forward = [AA(left_batch[i]) + AA(right_batch[i]) \
            for i in range(len(left_batch))]

    expected_backward = { 
            'left': ones_like(left_batch, PRECISION_TO_TYPE[precision]), 
            'right': ones_like(right_batch, PRECISION_TO_TYPE[precision]) 
            }

    left_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision]) for sample in left_batch]
    left_shape = left_value[0][0].shape
    right_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision]) for sample in right_batch]
    right_shape = right_value[0][0].shape

    a = I(shape=left_shape,
            data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
            needs_gradient=True,
            name='a')

    b = I(shape=right_shape,
            data_type=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
            needs_gradient=True,
            name='b')

    input_op_input = a + b
    forward_input = {a:left_value, b:right_value}
    output_shape = input_op_input.output().shape().dimensions()
    backward_input = { a: None, b: None }
    expected_backward = { a: expected_backward['left'], b: expected_backward['right'], }
    unittest_helper(input_op_input, 
        forward_input, expected_forward, 
        expected_backward,
        device_id, precision)

# -- minus operation tests --
#TODO: enable once the function is exposed
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_minus(left_operand, right_operand, device_id, precision):
    expected_forward = [AA([left_operand]) - AA([right_operand])]

    expected_backward = {
            'left_arg':  [[[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]]],
            'right_arg': [[[-1*np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in right_operand]]]
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
#TODO: enable once the function is exposed
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


# -- identity function tests --
# TODO enable this once the function is exposed
IDENTITY_TENSORS = [
    ([30.]),
    ([[30.]]),
    ([[1.5, 2.1]]),
    ([[100., 200.], [300., 400.], [10., 20.]]),
    ([[30,40], [1,2], [0.1, 0.2]])
]

@pytest.mark.parametrize("operand", IDENTITY_TENSORS)
def _test_op_identity(operand, device_id, precision):
    expected_forward = [AA([operand])]

    expected_backward = {
            'arg': np.ones_like(expected_forward),            
            }

    from cntk.ops import identity

    _test_unary_op(precision, device_id, identity, operand, 
        expected_forward, expected_backward)
