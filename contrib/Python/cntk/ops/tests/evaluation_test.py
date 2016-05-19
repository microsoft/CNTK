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
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from .. import dynamic_axis

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
    
    axis = dynamic_axis()
    target = I([target_vector], dynamic_axis=axis)
    output = I([output_vector], dynamic_axis=axis)
    
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

# -- SquareError with softmax operation tests --
@pytest.mark.parametrize("target_matrix, output_matrix", TARGET_OUT_PAIRS)
def test_op_square_error(target_matrix, output_matrix, device_id, precision):
    
    from .. import square_error

    def numpy_op(target, output): 
        return np.sum((target-output)**2)            
    
    axis = dynamic_axis()
    target = I([target_matrix], dynamic_axis=axis)
    output = I([output_matrix], dynamic_axis=axis)
    
    op_node = square_error(target, output)

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    expected = [[[numpy_op(AA(target_matrix, dtype=PRECISION_TO_TYPE[precision]), 
                              AA(output_matrix, dtype=PRECISION_TO_TYPE[precision]))]]]    
    unittest_helper(op_node, None, expected,
                device_id=device_id,
                precision=precision,
                clean_up=True, backward_pass=False)
                                
    def numpy_grad(left, right):                
        return 2*np.subtract(left, right)

    # Backward pass test
    # ==================
    # The expected results for the backward pass w.r.t. output is 2*(output - target)
    expected = [numpy_grad(AA(output_matrix, dtype=PRECISION_TO_TYPE[precision]), 
                              AA(target_matrix, dtype=PRECISION_TO_TYPE[precision]))]
    unittest_helper(op_node, None, expected,
            device_id=device_id,
            precision=precision, clean_up=True, backward_pass=True,
            input_node=output)
    
    # The expected results for the backward pass w.r.t. target is 2*(target - output)
    expected = [numpy_grad(AA(target_matrix, dtype=PRECISION_TO_TYPE[precision]), 
                              AA(output_matrix, dtype=PRECISION_TO_TYPE[precision]))]
    unittest_helper(op_node, None, expected,
            device_id=device_id,
            precision=precision, clean_up=True, backward_pass=True,
            input_node=target)

TARGET_OUT_PAIRS_EP = [
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),    
    ]

# -- ErrorPrediction with softmax operation tests --
@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS_EP)
def test_op_error_prediction(target_vector, output_vector, device_id, precision):
    
    from .. import error_prediction

    def numpy_op(target, output): 
        return np.argmax(target) != np.argmax(output)        
    
    axis = dynamic_axis()
    target = I([target_vector], dynamic_axis=axis)
    output = I([output_vector], dynamic_axis=axis)
    
    op_node = error_prediction(target, output)

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    expected = [[[numpy_op(AA(target_vector, dtype=PRECISION_TO_TYPE[precision]), 
                              AA(output_vector, dtype=PRECISION_TO_TYPE[precision]))]]]    
    unittest_helper(op_node, None, expected,
                device_id=device_id,
                precision=precision,
                clean_up=True, backward_pass=False)
