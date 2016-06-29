# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
import cntk as C

REDUCE_TEST_CASES = [
    #(input_data,  axis)
    ([1], 0),
    ([[1,2],[4,5]], 0),
    ([[1,2],[4,5]], 1),
    ([[1,2],[4,5]], 2),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], 0),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], 3),
]

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_CASES)
def test_op_reduce_sum(input_data, axis, device_id, precision):
    # Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # We need two surrounding brackets:
    # The first for sequences (length=1, since we have dynamic_axis='').
    # The second for batch of one sample.

    # keepdims = True as CNTK keeps them as well
    def reduce_sum(x, axis, keepdims=True):  
        x_aa = AA(x)
        if axis == len(x_aa.shape):
            return [np.reshape(np.add.reduce(np.ravel(x_aa)), (1,1))]
        return [[np.add.reduce(x_aa, axis, dtype=PRECISION_TO_TYPE[precision], 
                                                            keepdims=keepdims)]]
        
    expected_result = reduce_sum(input_data, axis)

    a = I([input_data])
        
    # splice using the operator
    result = C.reduce_sum(a, axis)

    unittest_helper(result, None, expected_result, device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The gradient of the reduce_sum operator is all ones in the shape of the input

    def grad_reduce_sum(x):
        return AA(np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]))

    expected_gradient = [[grad_reduce_sum(input_data)]]
        
    unittest_helper(result, None, expected_gradient, device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)




REDUCE_MAX_TEST_CASES = [
     ([[10, 0],[20,  1]], 2,        [20],  [[0,0],[1,0]]),
     ([[10, 0],[20,  1]], 1, [[10], [20]], [[1,0],[1,0]]),
     ([[10, 0],[ 0, 20]], 0, [[10,   20]], [[1,0],[0,1]]),
]

@pytest.mark.parametrize("input_data, axis_data, expected_result, expected_gradient", REDUCE_MAX_TEST_CASES)
def test_op_reduce_max(input_data, axis_data, expected_result, expected_gradient, device_id, precision):

    a = I([input_data])


    # slice using the operator
    result = C.reduce_max(a, axis = axis_data)

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)
    unittest_helper(result, None, [[expected_gradient]], device_id = device_id,
                precision=precision, clean_up=True, backward_pass=True, input_node=a)



REDUCE_MIN_TEST_CASES = [
     ([[-10, 0],[-20,  -1]], 2,         [-20],  [[0,0],[1,0]]),
     ([[-10, 0],[-20,  -1]], 1, [[-10], [-20]], [[1,0],[1,0]]),
     ([[-10,-0],[  0, -20]], 0, [[-10,   -20]], [[1,0],[0,1]]),
]

@pytest.mark.parametrize("input_data, axis_data, expected_result, expected_gradient", REDUCE_MIN_TEST_CASES)
def test_op_reduce_min(input_data, axis_data, expected_result, expected_gradient, device_id, precision):

    a = I([input_data])


    # slice using the operator
    result = C.reduce_min(a, axis = axis_data)

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)
    unittest_helper(result, None, [[expected_gradient]], device_id = device_id,
                precision=precision, clean_up=True, backward_pass=True, input_node=a)


