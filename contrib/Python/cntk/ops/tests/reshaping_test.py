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
    #(inputShape, outputShape, expectedOutputShape)
    ([2, 3],    [3, 2], [3, 2]),
    ([2, 3],    [6, 1], [6, 1]),
    ([2, 3],    [6, 1], [6, 1]),
    ([6, 1],    [2, 3], [2, 3]),
    ([2, 3, 5], [5, 6], [5, 6]),
    # now we test the feature that we can set one dimension of the outputShape to 0 meaning that it's value is inferred
    ([2, 3, 5], [0, 6], [5, 6]), 
    ([2, 3, 5], [5, 0], [5, 6]),
]

@pytest.mark.parametrize("inputShape, outputShape, expectedOutputShape", RESHAPE_TEST_CASES)
def test_op_reshape(inputShape, outputShape, expectedOutputShape, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
                        
    num_tensor_elements = np.multiply.reduce(inputShape)
    input_tensor = np.arange(num_tensor_elements).reshape(inputShape)
        
    expected_tensor = input_tensor.reshape(expectedOutputShape, order='F')

    a = I([input_tensor])

    # reshape into output shape
    reshaped_input = reshape(a, outputShape)

    unittest_helper(reshaped_input, None, [[expected_tensor]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we would compute the gradients on the unmodified reshape would would get 1 for all inputs.
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply the reshaping result with some weight tensor. 
    # For convienience choose '100 * expected_tensor' as weight.
    # The expected gradient is identical to this weight tensor reshaped according the input shape.

    a = I([input_tensor])

    # reshape into output shape
    reshaped_input = reshape(a, outputShape)

    some_factor = 100
    weight =  some_factor * expected_tensor
    output = reshaped_input * weight

    expected_gradient = input_tensor * some_factor 
    
    unittest_helper(output, None, [[expected_gradient]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)

