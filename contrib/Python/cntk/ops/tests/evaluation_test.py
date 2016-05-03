# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evaluation operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *

TARGET_OUT_PAIRS = [
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0.5, 0.5]], [[1., 2., 3., 4.]]),
    ([[0., 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
    ]

# -- CrossEntropy with softmax operation tests --
@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_crossentropywithsoftmax(target_vector, output_vector, device_id, precision):
    
    from .. import cross_entropy_with_softmax

    def numpy_softmax(x):
        x = AA(x, dtype=PRECISION_TO_TYPE[precision])
        ox = x - x.max()  # subtract max to avoid overflow

        expX = np.exp(ox)
        return expX / np.sum(expX)
    
    def numpy_op(label, softmax):
        return -np.sum(label * np.log(softmax, dtype=PRECISION_TO_TYPE[precision]), dtype=PRECISION_TO_TYPE[precision])
    
    target = I([target_vector], has_dynamic_axis=True)
    output = I([output_vector], has_dynamic_axis=True)
    
    op_node = cross_entropy_with_softmax(target, output)

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    expected = [[[numpy_op(AA(target_vector, dtype=PRECISION_TO_TYPE[precision]), numpy_softmax(output_vector))]]]
    unittest_helper(op_node, None, expected,
                device_id=device_id,
                precision=precision,
                clean_up=True, backward_pass=False)
                
                
    def numpy_grad(softmax, target):
        s = np.sum(target, dtype=np.double) #This should be 1.0
        return np.subtract(softmax * s , target)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is fi*sum(ti)-ti
    expected = [numpy_grad(numpy_softmax(output_vector), AA(target_vector, dtype=PRECISION_TO_TYPE[precision]))]
    unittest_helper(op_node, None, expected,
            device_id=device_id,
            precision=precision, clean_up=True, backward_pass=True,
            input_node=output)


