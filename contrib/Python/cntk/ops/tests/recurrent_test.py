# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for recurrent operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from .. import future_value, past_value, dynamic_axis

TENSORS = [
    # forward future_value results in [[4,5,6],[7,8,9],[10,11,12],[0.1,0.1,0.1]]
    # forward past_value results in [[0.1,0.1,0.1],[1,2,3],[4,5,6],[7,8,9]]
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 1, 0.1),

    # same as above but default value is different
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 1, 0.2),

    # now go 2 time_steps in the past/future, so we get:
    # future: [[7,8,9],[10,11,12],[0.1,0.1,0.1],[0.1,0.1,0.1]]
    # past: [[0.1,0.1,0.1],[0.1,0.1,0.1],[1,2,3],[4,5,6]]
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 2, 0.1),

    # and finally a different shape
    ([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], 1, 0.1),
]

# -- future_value tests --
@pytest.mark.parametrize("tensor, time_step, default_value", TENSORS)
def test_op_future_value(tensor, time_step, default_value, device_id, precision):    

    """
    This function shifts the tensor along its columns by `time_step`. It replaces 
    the "shifted-out" values with `default_value`.
    """
    def shift(x):
        x_shape = np.shape(x)        
        total_elements = x_shape[0] * x_shape[1]
        elements_to_roll = total_elements - (x_shape[1] * time_step)
        x = np.roll(AA(x, dtype=PRECISION_TO_TYPE[precision]), elements_to_roll)
        np.put(x, range(elements_to_roll, total_elements), default_value)
        return x

    #Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # A tensor of the same shape is expected, but shifted `time step`s into 
    # the future. If we get to the end, then the `default_hidden_activation` 
    # value is used for that entry.
    expected = [shift(tensor)]
    
    a = 0
    b = I([tensor], dynamic_axis=dynamic_axis())
    c = time_step    
    d = default_value  
    
    result = future_value(a, b, time_step=c, default_hidden_activation=d)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the future_value() function is equal to 1 everywhere 
    # with respect to the original input except for the column[s] that was[were] shifted 
    # out which will now contain zeros (we pass on the gradient for all other 
    # samples).
    expected = [[np.ones_like(x) for x in tensor]]
    expected[0][0:time_step] = [np.zeros_like(x) for x in expected[0][0:time_step]]

    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)
                    

# -- past_value tests --
@pytest.mark.parametrize("tensor, time_step, default_value", TENSORS)
def test_op_past_value(tensor, time_step, default_value, device_id, precision):    

    """
    This function shifts the tensor along its columns by -`time_step`. It replaces 
    the "shifted-out" values with `default_value`.
    """
    def shift(x):
        x_shape = np.shape(x)        
        elements_to_roll = x_shape[1] * time_step
        x = np.roll(AA(x, dtype=PRECISION_TO_TYPE[precision]), elements_to_roll)
        np.put(x, range(elements_to_roll), default_value)
        return x

    #Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # A tensor of the same shape is expected, but shifted `time step`s into 
    # the past. If we get to the start, then the `default_hidden_activation` 
    # value is used for that entry.
    expected = [shift(tensor)]
    
    a = 0
    b = I([tensor], dynamic_axis=dynamic_axis())
    c = time_step    
    d = default_value  
    
    result = past_value(a, b, time_step=c, default_hidden_activation=d)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the past_value() function is equal to 1 everywhere 
    # with respect to the original input except for the column[s] that was[were] shifted 
    # out which will now contain zeros (we pass on the gradient for all other 
    # samples).
    expected = [[np.ones_like(x) for x in tensor]]
    expected[0][-time_step:] = [np.zeros_like(x) for x in expected[0][0:time_step]]

    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)
