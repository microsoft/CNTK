# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation has one forward pass 
and one backward pass test
"""

import numpy as np
import pytest
from ...context import get_new_context
from ...graph import *
from ...reader import *

# keeping things short
C = constant
I = input
AA = np.asarray

# Testing inputs
@pytest.mark.parametrize("left_arg, right_arg", [
    ([30], [10]),
    ([[30]], [[10]]),
    ([[1.5,2.1]], [[10,20]]),
     #Adding two 3x2 inputs of sequence length 1
    ([[30,40], [1,2], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]), 
    ([5], [[30,40], [1,2]]),
    ])
def test_op_add_input_constant2(left_arg, right_arg):
    expected = AA(left_arg) + AA(right_arg)    
    # sequence of 1 element, since we have has_sequence_dimension=False
    expected = [expected] 
    # batch of one sample
    expected = [expected]    
    _test(I([left_arg], has_sequence_dimension=False) + right_arg, expected, False)
    _test(left_arg + I([right_arg], has_sequence_dimension=False), expected, False)
    
#TODO: move this method so it is used by other test files
def _test(root_node, expected, clean_up=True, backward_pass = False, input_node = None):
    """
    Helper functiuon for various operations unit tests
    """
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        assert not ctx.input_nodes
        result = ctx.eval(root_node, None, backward_pass, input_node)

        assert len(result) == len(expected)
        for res, exp in zip(result, expected):  
            assert np.allclose(res, exp)
            assert res.shape == AA(exp).shape