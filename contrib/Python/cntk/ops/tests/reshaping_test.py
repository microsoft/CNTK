# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from .. import reshape


RESHAPE_TEST_CASES = [
    #(inputShape, outputShape)
    ([2, 3], [3, 2]),
    ([2, 3], [6, 1]),
    ([2, 3], [1, 6]),
    ([6, 1], [2, 3]),
    ([2, 3, 5], [5, 6]),
]

#@pytest.mark.parametrize("inputShape, beginAxis, endAxis", RESHAPE_TEST_CASES)
@pytest.mark.parametrize("inputShape, outputShape", RESHAPE_TEST_CASES)
def test_op_reshape(inputShape, outputShape, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_dynamic_axis=False)
    # the second for batch of one sample
                        
    num_tensor_elements = np.multiply.reduce(inputShape)
    input_tensor = np.arange(num_tensor_elements).reshape(inputShape)
        
    expected_tensor = input_tensor.reshape(outputShape, order='F')

    a = I([input_tensor], has_dynamic_axis=False)

    # reshape into output shape
    reshaped_input = reshape(a, outputShape)

    unittest_helper(reshaped_input, None, [[expected_tensor]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # Reshaping is just mapping moving the input values to different to a different index in the out value.
    # 
    # For testing the gradients we want to have different gradients for each input value otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply the reshaping result with some weight tensor. 
    # For convienience choose '100 * expected_tensor' as weight.
    # The expected gradient is identical to this weight tensor reshaped according the input shape.

    a = I([input_tensor], has_dynamic_axis=False)

    # reshape into output shape
    reshaped_input = reshape(a, outputShape)

    some_factor = 100
    weight =  some_factor * expected_tensor
    output = reshaped_input * weight

    expected_gradient = input_tensor * some_factor 
    
    unittest_helper(output, None, [[expected_gradient]], device_id = device_id,
                    precision=precision, clean_up=False, backward_pass=True, input_node=a)

