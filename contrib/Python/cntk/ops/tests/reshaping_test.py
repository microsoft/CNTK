# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
import cntk as C

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
    reshaped_input = C.reshape(a, outputShape)

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
    reshaped_input = C.reshape(a, outputShape)

    some_factor = 100
    weight =  expected_tensor * some_factor

    output = reshaped_input * weight
    expected_gradient = input_tensor * some_factor 
    
    unittest_helper(output, None, [[expected_gradient]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)


SLICE_TEST_CASES = [
    #(input_data, slice_params(beg_index, end_index,axis), expected_result)
    ([[1,2],[-3,4]], (1,2,0), [[-3,4]]),
    ([[1,2],[-3,4]], (1,2,1), [[2],[4]]),
]
@pytest.mark.parametrize("input_data, slice_params, expected_result", SLICE_TEST_CASES)
def test_op_slice(input_data, slice_params, expected_result, device_id, precision):
    # Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # We need two surrounding brackets:
    # The first for sequences (length=1, since we have dynamic_axis='').
    # The second for batch of one sample.

    a = I([input_data])
    def op_slice(x, beg_index, end_index, axis):
        return x[beg_index:end_index]

    def _ax_slices(x, beg_index, end_index, axis):
        '''
        Creates a NumPy slicing array from slice operator's arguments
        '''
        ax_slices = []
        for i in range(0, len(x.shape)):
            if i==axis:
                if end_index >= x.shape[i]:
                    ax_slices.append([beg_index,])
                else:
                    ax_slices.append([beg_index,end_index])
            else:
                ax_slices.append(slice(None)) # corresponds to ':'
        return ax_slices


    # slice using the operator
    result = C.slice(a, *slice_params)

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # slice using the overload
    ax_slices = _ax_slices(a, *slice_params)
    result = a[ax_slices]

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=False, backward_pass=False)
    # Backward pass test
    # ==================
    # The gradient of the slice operator is a tensor of the same shape as the
    # input tensor, having 1 for elements that were taken and 0 for elements
    # that were dropped.

    def grad_slice(x, beg_index, end_index, axis):
        res = np.zeros_like(x)
        ax_slices = _ax_slices(x, beg_index, end_index, axis)
        res[ax_slices] = x[ax_slices]
        res[res!=0] = 1
        return res

    expected_gradient = grad_slice(np.asarray(input_data), *slice_params)
    
    unittest_helper(result, None, [[expected_gradient]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)

def test_op_slice_overload(device_id, precision):
    # Testing ComputationNode's __getitem__ more thoroughly

    input_data = np.arange(12).reshape(2,3,2)
    # array([[[ 0,  1],
    #         [ 2,  3],
    #         [ 4,  5]],
    #        [[ 6,  7],
    #         [ 8,  9],
    #         [10, 11]]])
    a = I([input_data])

    # simple index slicing
    result = a[1]

    expected_result = \
      np.asarray([[
                  [ 6,  7],
                  [ 8,  9],
                  [10, 11]]])
    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # slice a range along the middle axis
    result = a[:,1:,:]

    expected_result = \
      np.asarray([[
                  [ 2,  3],
                  [ 4,  5]],
                 [
                  [ 8,  9],
                  [10, 11]]])
    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # slice at the end
    result = a[:,:,1]

    expected_result = \
      np.asarray([[
                  [ 1],
                  [ 3],
                  [ 5]],
                 [[ 7],
                  [ 9],
                  [11]]])
    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # do we properly handle bad user input?
    with pytest.raises(ValueError):
        result = a[:,:,2:1]

    with pytest.raises(IndexError):
        result = a[1,object(),2]
