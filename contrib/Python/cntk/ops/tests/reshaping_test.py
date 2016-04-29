# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ..variables_and_parameters import *
from ...reader import *
from ..reshaping import reshape


RESHAPE_TEST_CASES = [
    #(inputShape, outputShape)
    ([2, 3], [3,2]),
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

    def make_tensor_with_shape(shape):

        def size_for_shape(shape):
            s = 1
            for dim in shape:
                s *= dim
            return s

        num_elements = size_for_shape(shape)
        data = [ x for x in range(0, num_elements)]
        tensor = np.array(data)
        tensor = tensor.reshape(shape)
        return tensor

    inputTensor    = make_tensor_with_shape(AA(inputShape))

    output_shape   = AA(outputShape);
    expectedTensor = inputTensor.reshape(output_shape)

    a = I([inputTensor], has_dynamic_axis=False)
    b = C(output_shape)
    result = reshape(a,tuple(output_shape))

    unittest_helper(result, None, expectedTensor, device_id=device_id, 
                precision=precision, clean_up=False, backward_pass=False)
