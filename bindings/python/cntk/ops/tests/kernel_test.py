# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for kernel operations, tested for the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, AA, I, precision, PRECISION_TO_TYPE, constant
from cntk.ops import AVG_POOLING, MAX_POOLING
from ...utils import sanitize_dtype_cntk, cntk_device

CONVOLUTION_OPERANDS = [
    ([[[5., 6.],  # (1, 2, 2) map
       [3., 4.]]],
     [[[1., 2.],  # (1, 2, 2) input operand
       [7., 8.]]]),
    ([[[1., 2.],  # (3, 2, 2) map
       [3., 4.]],
      [[1., 2.],
       [3., 4.]],
      [[1., 2.],
       [3., 4.]]],
     [[[1., 2.],  # (3, 2, 2) input operand
       [3., 4.]],
      [[5., 6.],
       [7., 8.]],
      [[9., 10.],
       [11., 12.]]])
]


@pytest.mark.parametrize("convolution_map, convolution_input", CONVOLUTION_OPERANDS)
def test_op_convolution_without_padding(convolution_map, convolution_input, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    conv_map = AA(convolution_map, dtype=dt)
    conv_input = AA(convolution_input, dtype=dt)

    flipped_conv_map = conv_map[..., ::-1, ::-1]

    from scipy import signal
    expected_forward = AA(
        [[signal.convolve(flipped_conv_map, conv_input, mode='valid')]])

    backward = AA(conv_map)

    a = I(shape=conv_input.shape,
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    conv_input.shape = (1,1) + conv_input.shape # adding batch and channel axis
    conv_map.shape = (1,1) + conv_map.shape

    constant_map = constant(value=conv_map, device=dev)

    from cntk import convolution
    input_op = convolution(constant_map, a, auto_padding=[False])

    forward_input = {a: conv_input}
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward,
                    expected_backward, device_id=device_id, precision=precision)

AVG_POOLING_DATA = [
    ([1, 2, 2, 4 ,3], # input_size
     (2, 2, 1), # pooling_window
     (2, 2, 1), # strides
     [[[[  8.5,   9.5,  10.5],
        [ 14.5,  15.5,  16.5]]],
      [[[ 32.5,  33.5,  34.5],
        [ 38.5,  39.5,  40.5]]]]), # result
    ([1, 1, 2, 2 ,4],
     (2, 2, 1),
     (2, 2, 1),
     [[[[  7.,   8.,   9.,  10.]]]])
]
@pytest.mark.parametrize("input_size, pooling_window, strides, result", AVG_POOLING_DATA)
def test_op_avg_pooling(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = I(shape=input_operand.shape[2:],
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    backward = (1 / np.prod(pooling_window)) * np.ones_like(input_operand)

    from cntk import pooling
    input_op = pooling(a, AVG_POOLING, pooling_window, strides, auto_padding=[True])

    forward_input = {a: input_operand}

    expected_forward = AA([result])
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward,
                expected_backward, device_id=device_id, precision=precision)

MAX_POOLING_DATA = [
    ([1, 2, 2, 4 ,3], # input_size
     (2, 2, 1), # pooling_window
     (2, 2, 1), # strides
     [[[[ 16.,  17.,  18.],
         [ 22.,  23.,  24.]]],
       [[[ 40.,  41.,  42.],
         [ 46.,  47.,  48.]]]]), # result

    ([1, 2, 4, 4 ,4],
     (2, 2, 2),
     (2, 2, 2),
     [[[[  22.,   24.],
        [  30.,   32.]],
       [[  54.,   56.],
        [  62.,   64.]]],
      [[[  86.,   88.],
        [  94.,   96.]],
       [[ 118.,  120.],
        [ 126.,  128.]]]]),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, result", MAX_POOLING_DATA)
def test_op_max_pooling(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = I(shape=input_operand.shape[2:],
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    result_array = np.asarray(result, dtype=dt)
    max_elements = result_array.reshape(result_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(input_operand)
    for element in max_elements:
        backward += np.asarray(input_operand == element)

    from cntk import pooling
    input_op = pooling(a, MAX_POOLING, pooling_window, strides)

    forward_input = {a: input_operand}

    expected_forward = AA([result])
    expected_backward = {a: backward}

    unittest_helper(input_op,
                forward_input, expected_forward, expected_backward,
                device_id=device_id, precision=precision)

# ROI pooling test setup
# --- forward ---
# input convFeatureMap 3x3 map, values [[1,2,3][4,5,6][7,8,9]]
# input rois 4x1, values (x, y, w, h) = (1/3, 1/3, 2/3, 2/3)
# roiOutputShape 3 x 3
# expected output 3x3 map, values [[5,6,6][8,9,9][8,9,9]]
# --- backward ---
# gradient 3x3 map, values [[1,1,1][1,1,1][1,1,1]]
# expected output gradient 3x3 map, values [[0,0,0][0,1,2][0,2,4]]
ROIPOOLING_OPERANDS = [
    ([[[1., 2., 3.],       # (1, 3, 3) input operand (conv feature map)
       [4., 5., 6.],
       [7., 8., 9.]]],
     [[.33, .33, .66, .66]], # (4) input roi (x, y, w, h) relative to image width and height
     [[[5., 6., 6.],       # (1, 3, 3) expected forward output
       [8., 9., 9.],
       [8., 9., 9.]]],
     [[[0., 0., 0.],       # (1, 3, 3) expected backward output (gradient input is all 1s)
       [0., 1., 2.],
       [0., 2., 4.]]])
]

@pytest.mark.parametrize("input_map, input_rois, expected_fwd, expected_bkwd", ROIPOOLING_OPERANDS)
def test_op_roipooling(input_map, input_rois, expected_fwd, expected_bkwd, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # AA == as numpy array
    conv_input        = AA(input_map, dtype=dt)
    roi_input         = AA(input_rois, dtype=dt)
    exp_fwd_value     = AA(expected_fwd, dtype=dt)
    exp_bkwd_value    = AA(expected_bkwd, dtype=dt)
    
    # adding batch, sequence and roi axis
    exp_fwd_value.shape  = (1,1,1) + exp_fwd_value.shape
    exp_bkwd_value.shape = (1,1) + exp_bkwd_value.shape

    # I == define cntk input variables
    a = I(shape=conv_input.shape,
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    b = I(shape=roi_input.shape,
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=False,
        name='b')

    # adding batch and sequence axis
    conv_input.shape     = (1,1) + conv_input.shape
    roi_input.shape      = (1,1) + roi_input.shape
    
    from cntk import roipooling
    input_op = roipooling(a, b, (3,3))

    forward_input = {a: conv_input, b: roi_input}
    expected_backward = {a: exp_bkwd_value}

    unittest_helper(input_op,
                    forward_input, exp_fwd_value, expected_backward,
                    device_id=device_id, precision=precision)
