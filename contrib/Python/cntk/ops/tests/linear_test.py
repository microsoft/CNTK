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
from .ops_test_utils import unittest_helper, AA, I, SI, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type
from ...graph import *
from .. import *
from ...reader import *
import numpy as np

# TODO: Test plus(), times(), etc, not only the overloaded opeartors (+,
# *, etc.)

# TODO: perhaps include some rand() testing; and
TENSOR_PAIRS = [
    ([30.], [10.]),
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10., 20.]]),
    ([[100., 200.], [300., 400.], [10., 20.]],
     [[10., 20.], [30., 40.], [1., 2.]]),
    # Test with broadcast
    # TODO: adjust the broadcast according to the decision on col/row major
    #([5], [[10, 20], [30,40], [1,2]]),     
    
    # Adding two 3x2 inputs of sequence length 1
    ([[30.,40.], [1.,2.], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]),
]

# -- plus operation tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_plus(left_operand, right_operand, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) + AA(right_operand, dtype=PRECISION_TO_TYPE[precision])]]

    a = I([left_operand])
    b = I([right_operand])

    left_as_input = a + right_operand
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    right_as_input = left_operand + b
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(a + b, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    #==================
    # the expected results for the backward pass is all ones

    expected = [[[np.ones_like(x) for x in left_operand]]]    
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)
    
    expected = [[[np.ones_like(x) for x in right_operand]]]
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)

# -- minus operation tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_minus(left_operand, right_operand, device_id, precision):

    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) - AA(right_operand, dtype=PRECISION_TO_TYPE[precision])]]

    a = I([left_operand])
    b = I([right_operand])

    left_as_input = a - right_operand
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    right_as_input = left_operand - b
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(a - b, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    #==================
    # the expected results for the backward pass is all ones for left and
    # negative ones for right operand
    expected = [[[np.ones_like(x) for x in left_operand]]]
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)
    expected = [[[-np.ones_like(x) for x in right_operand]]]
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)

# -- element times tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_times(left_operand, right_operand, device_id, precision):

    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) * AA(right_operand, dtype=PRECISION_TO_TYPE[precision])]]

    a = I([left_operand])
    b = I([right_operand])

    left_as_input = a * right_operand
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    right_as_input = left_operand * b
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(a * b, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    #==================

    # for element_times the derivative is left wrt right, and right wrt left
    expected_left = [[right_operand]]
    expected_right = [[left_operand]]

    unittest_helper(left_as_input, None, expected_left, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)
    unittest_helper(right_as_input, None, expected_right, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)

# -- element divide tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_divide(left_operand, right_operand, device_id, precision):

    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) / AA(right_operand, dtype=PRECISION_TO_TYPE[precision])]]

    a = I([left_operand])
    b = I([right_operand])

    left_as_input = a / right_operand
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    right_as_input = left_operand / b
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(a / b, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    #==================
    # For left: d/da (a/b) = 1/b
    # For right: d/db (a/b) = a * d/db (1/b) = a * -1/b^2 = -a/b^2
    expected_left = [[[np.ones_like(x) / x for x in right_operand]]]
    expected_right = [[-AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) / AA(right_operand, dtype=PRECISION_TO_TYPE[precision])**2]]

    unittest_helper(left_as_input, None, expected_left, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)
    unittest_helper(right_as_input, None, expected_right, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)

IDENTITY_TENSORS = [
    ([30.]),
    ([[30.]]),
    ([[1.5, 2.1]]),
    ([[100., 200.], [300., 400.], [10., 20.]]),
    ([[30,40], [1,2], [0.1, 0.2]])
]

# -- identity function tests --

@pytest.mark.parametrize("tensor", IDENTITY_TENSORS)
def test_op_identity(tensor, device_id, precision):

    def numpy_op(x):
        return AA(x, dtype=PRECISION_TO_TYPE[precision])

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = identity(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass of x is x
    expected = np.ones_like(expected)

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


TIMES_PAIRS = [
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10.], [20.]]),
    ([[100., 200.]], [[10.], [20.]]),
]

@pytest.mark.parametrize("left_operand, right_operand", TIMES_PAIRS)
def test_op_times(left_operand, right_operand, device_id, precision,
        left_matrix_type, right_matrix_type):
    if right_matrix_type == 'sparse':
        pytest.skip('second operator of times() has to be dense')

    dt = PRECISION_TO_TYPE[precision]
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[np.dot(AA(left_operand, dtype=dt), AA(right_operand, dtype=dt))]]
    
    if left_matrix_type == 'sparse':
        a = SI(*batch_dense_to_sparse([left_operand]))
    else:
        a = I([left_operand])

    b = I([right_operand])

    from cntk.ops import times, constant
    left_as_input = times(a, constant(right_operand))
    right_as_input = times(constant(left_operand), b)

    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(times(a, b), None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)


    # Backward pass test
    #==================

    def op_grad(A, B):
        '''
        Compute derivative of A with respect to B. For simplicity, assume A
        and B to be matrices.
        Let A be 2x2 and B be 2x1, then we have
        [a11 a12] [b11]  = [ a11 b11 + a12 b21 ]
        [a21 a22] [b21]    [ a21 b11 + a22 b21 ]

        The derivative for A with respect to B is
        [b11 b21]
        [b11 b21]

        The derivative for B with respect to A:
        [a11 + a12]
        [a21 + a22]
        '''
        assert len(A.shape) == len(B.shape) == 2
        D = np.zeros_like(A)
        D[:,:] = B.sum(axis=1)
        
        return D

    if 'sparse' not in [left_matrix_type, right_matrix_type]:
        # FIXME: disabling until the Pass node supports sparse 
        expected_left = [[op_grad(AA(left_operand, dtype=dt), AA(right_operand, dtype=dt))]]
        expected_right = [[op_grad(AA(right_operand, dtype=dt).T, AA(left_operand, dtype=dt).T).T]]

        unittest_helper(left_as_input, None, expected_left, device_id=device_id,
                        precision=precision, clean_up=True, backward_pass=True, input_node=a)
        # BUG: Fails because of Pass node?
        unittest_helper(right_as_input, None, expected_right, device_id=device_id,
                        precision=precision, clean_up=True, backward_pass=True, input_node=b)
