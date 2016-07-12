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
    #(input_shape, output_shape, expected_output_shape)
    ([2, 3],    [3, 2], [3, 2]),
    ([2, 3],    [6, 1], [6, 1]),
    ([2, 3],    [6, 1], [6, 1]),
    ([6, 1],    [2, 3], [2, 3]),
    ([2, 3, 5], [5, 6], [5, 6]),
    # now we test the feature that we can set one dimension of the output_shape to 0 meaning that it's value is inferred
    ([2, 3, 5], [0, 6], [5, 6]), 
    ([2, 3, 5], [5, 0], [5, 6]),
]

@pytest.mark.parametrize("input_shape, output_shape, expected_output_shape", RESHAPE_TEST_CASES)
def test_op_reshape(input_shape, output_shape, expected_output_shape, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
                        
    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(num_tensor_elements).reshape(input_shape)
        
    expected_tensor = input_tensor.reshape(expected_output_shape, order='C')

    a = I([input_tensor])

    # reshape into output shape

    reshaped_input = C.reshape(a, output_shape)

    unittest_helper(reshaped_input, None, [[expected_tensor]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we compute the gradients on the unmodified tensor, reshape would get 1 for all inputs.
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply the reshaping result with itself.     
    # The expected gradient is identical to the input tensor.

    a = I([input_tensor])

    # reshape into output shape
    reshaped_input = C.reshape(a, output_shape)

    output = reshaped_input * expected_tensor

    unittest_helper(output, None, [[input_tensor]], device_id = device_id,
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
                precision=precision, clean_up=True, backward_pass=False)
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

SLICE_SEQ_TEST_CASES = [
    #(input_data, slice_params(beg_index, end_index,axis), expected_result)
    ([[[1,2,3]],[[-4,5,6]],[[7,8,9]]], (0,2), [[[1,2,3]],[[-4,5,6]]]),
    ([[[1,2,3],[11,12,13]],[[-4,5,6],[-14,15,16]],[[7,8,9],[17,18,19]]], 
        (0,2), [[[1,2,3],[11,12,13]],[[-4,5,6],[-14,15,16]]]),
    ([[[1,2,3],[11,12,13]],[[-4,5,6],[-14,15,16]],[[7,8,9],[17,18,19]]], 
        (1,2), [[[-4,5,6],[-14,15,16]]]),
]
@pytest.mark.parametrize("input_data, slice_params, expected_result", SLICE_SEQ_TEST_CASES)
def test_op_slice_sequence(input_data, slice_params, expected_result, device_id, precision):
    # Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # We need two surrounding brackets:
    # The first for sequences (length=1, since we have dynamic_axis='').
    # The second for batch of one sample.

    # 1 sample with 2 sequence element of a vector of 3

    t = C.dynamic_axis(name='t')
    a = I([input_data], dynamic_axis=t)

    # slice using the operator
    result = C.slice(a, slice_params[0], slice_params[1], axis='t')
    result = C.identity(result) # required hack because Slice doesn't propagate tag

    unittest_helper(result, None, [expected_result], device_id=device_id, 
                precision=precision, clean_up=False, backward_pass=False)

    # Backward pass test
    # ==================
    # The gradient of the slice operator is a tensor of the same shape as the
    # input tensor, having 1 for elements that were taken and 0 for elements
    # that were dropped.

    def grad_slice(x, beg_index, end_index):
        res = np.zeros_like(x)
        res[beg_index:end_index] = 1
        return res

    expected_gradient = grad_slice(np.asarray(input_data), *slice_params)
    
    unittest_helper(result, None, [expected_gradient], device_id = device_id,
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


SPLICE_TEST_CASES = [
    #(input_data1, input_data2, axis, expected_result)
    ([1], [2], 0, [1,2]),
    ([[1,2],[4,5]], [[10,20],[30, 40],[50, 60]], 0, 
     [[1, 2],[4, 5],[10, 20],[30, 40],[50, 60]]),
    ([[1,2],[4,5]], [[10,20,30],[40, 50, 60]], 1, 
     [[1,2,10,20,30],[4,5,40,50,60]]),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], [[10,20],[30,40]], 0, 
     [[[1,2],[3,4]],[[5,6],[7,8]],[[10,20],[30,40]]]),    
]
@pytest.mark.parametrize("input_data1, input_data2, axis, expected_result", SPLICE_TEST_CASES)
def test_op_splice(input_data1, input_data2, axis, expected_result, device_id, precision):
    # Forward pass test
    #==================
    # We compute the expected output for the forward pass.
    # We need two surrounding brackets:
    # The first for sequences (length=1, since we have dynamic_axis='').
    # The second for batch of one sample.

    a = I([input_data1])
    b = I([input_data2])
    
    # splice using the operator
    result = C.splice((a, b), axis)

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The gradient of the splice operator is all ones in the shape of the input

    def grad_splice(x):
        return np.ones_like(x)

    expected_gradient1 = grad_splice(np.asarray(input_data1))
    expected_gradient2 = grad_splice(np.asarray(input_data2))
    
    unittest_helper(result, None, [[expected_gradient1]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)

    unittest_helper(result, None, [[expected_gradient2]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)


TRANSPOSE_DIMS_TEST_CASES = [
    #(input_shape, axis1, axis2, expected_output_shape)
    ([2, 3],     0, 1, [3, 2]),
    ([2, 3],    1, 0, [3, 2]),    
    ([2, 3, 5], 0, 2, [5, 3, 2]), 
    ([2, 2, 2], 0, 1, [2, 2, 2]),
]

@pytest.mark.parametrize("input_shape, axis1, axis2, expected_output_shape", TRANSPOSE_DIMS_TEST_CASES)
def test_op_transpose_dimensions(input_shape, axis1, axis2, expected_output_shape, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
                        
    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(num_tensor_elements).reshape(input_shape)
    
    permutated_axes = np.arange(len(input_shape))
    axis1_idx = permutated_axes[axis1]
    permutated_axes[axis1] = permutated_axes[axis2]
    permutated_axes[axis2] = axis1_idx    
    expected_tensor = input_tensor.transpose(*permutated_axes)
    
    a = I([input_tensor])

    # swap two axes
    reshaped_input = C.transpose_dimensions(a, axis1, axis2)

    unittest_helper(reshaped_input, None, [[expected_tensor]], device_id=device_id, 
                precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # Reshaping is just moving the input values to different indexes of the result tensor.
    # If we compute the gradients on the unmodified tensor, reshape would get 1 for all inputs.
    # For testing the gradients we want to have different gradients for each input index otherwise we can't
    # test if they get wrongly permuted during test. To this end we multiply the reshaping result with itself.     
    # The expected gradient is identical to the input tensor.

    a = I([input_tensor])

    # swap two axes
    reshaped_input = C.transpose_dimensions(a, axis1, axis2)

    output = reshaped_input * expected_tensor            
    
    unittest_helper(output, None, [[input_tensor]], device_id = device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)

