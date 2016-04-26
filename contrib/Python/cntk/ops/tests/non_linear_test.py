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
from ..non_linear import clip, cond

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

# -- clip operation tests --
@pytest.mark.parametrize("min_value, max_value, x", CLIP_TUPLES)
def test_op_clip(min_value, max_value, x, device_id, precision):    

#Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # Compare to numpy's implementation of np.clip(x, min, max)
    expected = [[np.clip(AA(x, dtype=PRECISION_TO_TYPE[precision]), AA(min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))]]
    
    a = C(min_value)    
    b = C(max_value)
    c = I([x], has_sequence_dimension=False)
    
    result = clip(a, b, c)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the clip() function is equal to 1 when the element 
    # has not been clipped, and 0 if it has been clipped
    expected = [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]

    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=c)






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
