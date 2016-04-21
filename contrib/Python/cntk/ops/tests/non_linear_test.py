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
from ..non_linear import clip

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
