# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision
from ...graph import *
from ...reader import *
import numpy as np

#TODO: Test plus(), times(), etc, not only the overloaded opeartors (+, *, etc.)

#TODO: perhaps include some rand() testing; and 
TENSOR_PAIRS = [
    ([30], [10]),
    ([[30]], [[10]]),
    ([[1.5,2.1]], [[10,20]]),
    ([[100, 200], [300,400], [10,20]], [[10, 20], [30,40], [1,2]]),
    #Test with broadcast
    #TODO: un-expected CNTK output with the backward pass test
    #([5], [[10, 20], [30,40], [1,2]]),
    #TODO: enable once all branches are merged to master
    #Adding two 3x2 inputs of sequence length 1
    #([[30,40], [1,2], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]),     
    ]

# -- plus operation tests --
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_plus(left_operand, right_operand, device_id, precision):    

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    expected = [[AA(left_operand) + AA(right_operand)]]

    a = I([left_operand], has_sequence_dimension=False)
    b = I([right_operand], has_sequence_dimension=False)    
    
    left_as_input = a + right_operand    
    unittest_helper(left_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    right_as_input = left_operand + b
    unittest_helper(right_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    #the expected results for the backward pass is all ones
    expected = [[[np.ones_like(x) for x in left_operand]]]
    unittest_helper(left_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)    
    unittest_helper(right_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)    

# -- element times tests --
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_times(left_operand, right_operand, device_id, precision):

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    expected = [[AA(left_operand) * AA(right_operand)]]

    a = I([left_operand], has_sequence_dimension=False)
    b = I([right_operand], has_sequence_dimension=False)    
    
    left_as_input = a * right_operand    
    unittest_helper(left_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    right_as_input = left_operand * b
    unittest_helper(right_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
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

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    expected = [[AA(left_operand) / AA(right_operand)]]

    a = I([left_operand], has_sequence_dimension=False)
    b = I([right_operand], has_sequence_dimension=False)    
    
    left_as_input = a / right_operand    
    unittest_helper(left_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    right_as_input = left_operand / b
    unittest_helper(right_as_input, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # For left: d/da (a/b) = 1/b
    # For right: d/db (a/b) = a * d/db (1/b) = a * -1/b^2 = -a/b^2
    expected_left = [[[np.ones_like(x) / x for x in right_operand]]]
    expected_right = [[-AA(left_operand)  / AA(right_operand)**2]]
    
    unittest_helper(left_as_input, None, expected_left, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)    
    unittest_helper(right_as_input, None, expected_right, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)       
