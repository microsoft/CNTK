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
from .ops_test_utils import unittest_helper, C, AA, I, precision
from ...graph import *
from ...reader import *
from ..non_linear import clip_by_value
import numpy as np

CLIP_TUPLES = [
    ([1.5], [1.0], [2.0]), # value shouldn't be clipped
    ([0.5], [1.0], [2.0]), # value should be clipped to 1.0
    ([2.5], [1.0], [2.0]), # value should be clipped to 2.0
    ([[1.5, 2.1, 0.9]], [1.0], [2.0]), # should clip to [1.5, 2.0, 1.0]
    # should clip to [[1.0, 2.0], [1.0, 2.0], [1.5, 2.0]]    
    ([[0.0, 3.0], [1.0, 2.0], [1.5, 2.5]], [1.0], [2.0]),
    ]

# -- clip_by_value operation tests --
@pytest.mark.parametrize("x, min_value, max_value", CLIP_TUPLES)
def test_op_clip_by_value(x, min_value, max_value, device_id, precision):    

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # Compare to numpy's implementation of clip()
    expected = [[np.clip(AA(x), AA(min_value), AA(max_value))]]

    a = I([x], has_sequence_dimension=False)
    b = I([min_value], has_sequence_dimension=False)    
    c = I([max_value], has_sequence_dimension=False)
    
    result = clip_by_value(a, b, c)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=False, backward_pass=False)
    
    #Backward pass test
    #==================
    # ...
    #expected = [[[np.ones_like(x) for x in left_operand]]]
    #unittest_helper(left_as_input, None, expected, device_id=device_id, 
    #                precision=precision, clean_up=True, backward_pass=True, input_node=a)    
    #unittest_helper(right_as_input, None, expected, device_id=device_id, 
    #                precision=precision, clean_up=True, backward_pass=True, input_node=b)    