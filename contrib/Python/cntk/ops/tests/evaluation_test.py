# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evalation operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from ..evaluation import crossentropy_with_softmax

TARGET_OUT_PAIRS = [
    ([[0.0, 0.0, 0.5, 0.5]], [[1., 2., 3., 4.]]),
    ([[0.0, 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
    ]

@pytest.mark.parametrize("target_values, feature_values", TARGET_OUT_PAIRS)
def test_op_crossentropywithsoftmax(target_values, feature_values, device_id, precision):
    
    def numpy_softmax(x):
        x = AA(x, dtype=PRECISION_TO_TYPE[precision])
        ox = x - x.max()  # subtract max to avoid overflow

        expX = np.exp(ox)
        return expX / np.sum(expX)
    
    def numpy_op(label, softmax):
        return -np.sum(label * np.log(softmax, dtype=PRECISION_TO_TYPE[precision]), dtype=PRECISION_TO_TYPE[precision])
    
    
    input_target = I([target_values], has_dynamic_axis=True)
    input_features = I([feature_values], has_dynamic_axis=True)
    
    op_node = crossentropy_with_softmax(input_target, input_features)

    expected = [[[numpy_op(AA(target_values, dtype=PRECISION_TO_TYPE[precision]), numpy_softmax(feature_values))]]]
    
    unittest_helper(op_node, None, expected,
                device_id=device_id,
                precision=precision,
                clean_up=False, backward_pass=False)
                
                
    def numpy_grad(softmax, target):
        s = np.sum(target, dtype=np.double) #This should be 1.0
        return np.subtract(softmax * s , target)
        
    expected = [numpy_grad(numpy_softmax(feature_values), AA(target_values, dtype=PRECISION_TO_TYPE[precision]))]


    unittest_helper(op_node, None, expected,
            device_id=device_id,
            precision=precision, clean_up=True, backward_pass=True,
            input_node=input_features)
