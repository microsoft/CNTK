# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from ..non_linear import cond
import numpy as np

COND_TUPLES = [ 
						([-1], [2], [3]), 
                        ([0], [20], [30]),
						([10],[0],[-100]),
					  ]
					  
# -- cond operation tests --
@pytest.mark.parametrize("flag, value_a, value_b", COND_TUPLES)
def test_op_cond(flag, value_a, value_b, device_id, precision):    

    #Forward pass test
    #==================
    # Comparing to numpy's implementation of where(...)

    expected = [[[np.where(AA(flag, dtype=PRECISION_TO_TYPE[precision]), AA(value_a, dtype=PRECISION_TO_TYPE[precision]), AA(value_b, dtype=PRECISION_TO_TYPE[precision]))]]]

    cond_as_const    = C([flag])
    value_a_as_const = C([value_a])    
    value_b_as_const = C([value_b])   
	
    cond_as_input    = I([flag],    has_sequence_dimension=False)
    value_a_as_input = I([value_a], has_sequence_dimension=False)
    value_b_as_input = I([value_b], has_sequence_dimension=False)
	
    result = cond(cond_as_input, value_a_as_const, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
					
	#Backward pass test
    #==================
    # The derivative of the cond() function is zero for the first argument.
	# The derivative for second and thrird argument depends on the first:
	# * Derivative of second argument = derivative of input if cond else 0
	# * Derivative of third argument  = derivative of input if not cond else 0

    # Derivative for first parameter should always be zero
    expected  = [[[np.zeros_like(x) for x in flag]]]
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=cond_as_input)

    # Derivative of second parameter depends on cond
    expected = [[np.array(np.where(flag, 1, 0), dtype=PRECISION_TO_TYPE[precision])]]
    result = cond(cond_as_const, value_a_as_input, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_a_as_input)

    # Derivative of third parameter depends on cond
    expected = [[np.array(np.where(flag, 0, 1), dtype=PRECISION_TO_TYPE[precision])]]
    result = cond(cond_as_const, value_a_as_const, value_b_as_input)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_b_as_input)