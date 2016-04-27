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
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from ..recurrent import future_value, past_value

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
]

# -- future_value tests tests --
@pytest.mark.parametrize("tensor, time_step, default_value", TENSORS)
def test_op_future_value(tensor, time_step, default_value, device_id, precision):    

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
    b = I([tensor], has_dynamic_axis=True)
    c = time_step    
    d = default_value  
    
    result = future_value(a, b, time_step=c, default_hidden_activation=d)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=False, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the clip() function is equal to 1 when the element 
    # has not been clipped, and 0 if it has been clipped
    #expected = [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]

    #unittest_helper(result, None, expected, device_id=device_id, 
    #                precision=precision, clean_up=True, backward_pass=True, input_node=c)