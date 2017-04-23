# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for kernel operations, tested for the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
import cntk as C
from .ops_test_utils import unittest_helper, _test_unary_op, AA, precision, PRECISION_TO_TYPE, constant, cntk_device
from cntk.ops import AVG_POOLING, MAX_POOLING, MAX_UNPOOLING
from cntk.internal import sanitize_dtype_cntk

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
    expected_forward = AA([signal.convolve(flipped_conv_map, conv_input, mode='valid')])

    backward = AA(conv_map)

    a = C.input(shape=conv_input.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=True,
                name='a')

    conv_input.shape = (1,) + conv_input.shape # adding batch and channel axis
    conv_map.shape = (1,) + conv_map.shape

    constant_map = constant(value=conv_map, device=dev)

    from cntk import convolution
    input_op = convolution(constant_map, a, auto_padding=[False])

    forward_input = {a: conv_input}
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward,
                    expected_backward, device_id=device_id, precision=precision)


ASYM_CONVOLUTION_DATA = [
    ([1, 1, 3, 3], # input_size
     [1, 2, 2], # convolution size
     [[[[ 19, 25, 10],
        [ 37, 43, 16],
        [ 7, 8, 0]]]]) # result
]
# this test handles convolution with asymmetric padding, in particular, with auto_padding is set to True
# and the kernel shape is even
@pytest.mark.skip(reason="Reference model takes too long to run causing timeout, needs further investigation")
@pytest.mark.parametrize("input_size, conv_size, result", ASYM_CONVOLUTION_DATA)
def test_asym_convolution(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution
    input_op = convolution(conv_map, a, auto_padding=[True])

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)


POOLING_GEOMETRY_DATA = [
    ([1, 1, 6, 6], # input_size
     (1, 5, 5), # pooling_window
     (1, 3, 3), # strides
     [True], # padding flag
     [[[[ 21,   23],
        [ 33,   35]]]]), # result
    ([1, 1, 8, 8],
     (1, 4, 4),
     (1, 5, 5),
     [False],
     [[[[ 27 ]]]]),
    ([1, 1, 6, 6],
     (1, 4, 4),
     (1, 2, 2),
     [True, False],
     [[[[ 15, 17],
        [ 27, 29],
        [ 33, 35]]]])
]
# the pooling geometry test also tests convolution geometry since they go through the same path
# in the CPU code
@pytest.mark.parametrize("input_size, pooling_window, strides, padding, result", POOLING_GEOMETRY_DATA)
def test_op_pooling_geometry(input_size, pooling_window, strides, padding, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    from cntk import pooling
    input_op = pooling(a, MAX_POOLING, pooling_window, strides, auto_padding=padding)

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)

AVG_POOLING_DATA = [
    ([1, 2, 2, 4, 3], # input_size
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

    a = C.sequence.input(shape=input_operand.shape[2:],
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
    ([1, 2, 2, 4, 3], # input_size
     (2, 2, 1), # pooling_window
     (2, 2, 1), # strides
     [False],   # autopad
     [[[[ 16.,  17.,  18.],
         [ 22.,  23.,  24.]]],
       [[[ 40.,  41.,  42.],
         [ 46.,  47.,  48.]]]]), # result

    ([1, 2, 4, 4, 4],
     (2, 2, 2),
     (2, 2, 2),
     [False],
     [[[[  22.,   24.],
        [  30.,   32.]],
       [[  54.,   56.],
        [  62.,   64.]]],
      [[[  86.,   88.],
        [  94.,   96.]],
       [[ 118.,  120.],
        [ 126.,  128.]]]]),

    ([1, 1, 1, 8, 8],
     (5, 5),
     (2, 2),
     [True],
     [[[[ 19.,  21.,  23.,  24.],
        [ 35.,  37.,  39.,  40.],
        [ 51.,  53.,  55.,  56.],
        [ 59.,  61.,  63.,  64.]]]])
]


@pytest.mark.parametrize("input_size, pooling_window, strides, autopad, result", MAX_POOLING_DATA)
def test_op_max_pooling(input_size, pooling_window, strides, autopad, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.sequence.input(shape=input_operand.shape[2:],
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
    input_op = pooling(a, MAX_POOLING, pooling_window, strides, autopad)

    forward_input = {a: input_operand}

    expected_forward = AA([result])
    expected_backward = {a: backward}

    unittest_helper(input_op,
                forward_input, expected_forward, expected_backward,
                device_id=device_id, precision=precision)


@pytest.mark.parametrize("input_size, pooling_window, strides, autopad, result", MAX_POOLING_DATA)
def test_op_max_unpooling(input_size, pooling_window, strides, autopad, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]


    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.sequence.input(shape=input_operand.shape[2:],
                         dtype=sanitize_dtype_cntk(precision),
                         needs_gradient=True,
                         name='a')

    pooling_result = np.asarray(result, dtype=dt)
    max_elements = pooling_result.reshape(pooling_result.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(input_operand)
    for element in max_elements:
        backward += np.asarray(input_operand == element)

    from cntk import pooling, unpooling
    p = pooling(a, MAX_POOLING, pooling_window, strides, autopad)
    u = unpooling(p, a, MAX_UNPOOLING, pooling_window, strides, autopad)
    q = pooling(u, MAX_POOLING, pooling_window, strides, autopad)

    forward_input = {a: input_operand}

    expected_forward = backward * input_operand
    expected_backward = {a: backward}

    unittest_helper(u,
                forward_input, expected_forward, expected_backward,
                device_id=device_id, precision=precision)
    assert np.allclose(p.eval(forward_input), q.eval(forward_input))

POOLING_CEIL_DATA = [
    ([1, 1, 8, 8],                   # input_size
     (2, 2),                            # pooling_window
     (2, 2),                            # strides
     [[[[10.,  12.,  14.,  16.],
        [26.,  28.,  30.,  32.],
        [42.,  44.,  46.,  48.],
        [58.,  60.,  62.,  64.]]]]),    # result
    ([1, 1, 8, 8],
     (3, 3),
     (2, 2),
     [[[[19., 21., 23., 24.],
        [35., 37., 39., 40.],
        [51., 53., 55., 56.],
        [59., 61., 63., 64.]]]]),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, result", POOLING_CEIL_DATA)
def test_op_pooling_ceil(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:], dtype=sanitize_dtype_cntk(precision), needs_gradient=True, name='a')

    result_array = np.asarray(result, dtype=dt)
    max_elements = result_array.reshape(result_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(input_operand)
    for element in max_elements:
        backward += np.asarray(input_operand == element)

    from cntk import pooling
    input_op = pooling(a, MAX_POOLING, pooling_window, strides, ceil_out_dim=True)

    forward_input = {a: input_operand}

    expected_forward = AA(result)
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward, expected_backward, device_id=device_id,
                    precision=precision)

POOLING_AVG_INCLUDE_PAD_DATA = [
    ([1, 1, 7, 7],
     (3, 3),
     (3, 3),
     [[[[20./9, 45./9, 40./9],
        [135./9, 225./9, 165./9],
        [160./9, 255./9, 180./9]]]]),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, result", POOLING_AVG_INCLUDE_PAD_DATA)
def test_op_average_pooling_include_pad(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:], dtype=sanitize_dtype_cntk(precision), needs_gradient=True, name='a')

    backward = (1 / np.prod(pooling_window)) * np.ones_like(input_operand)

    from cntk import pooling
    input_op = pooling(a, AVG_POOLING, pooling_window, strides, auto_padding=[True], include_pad=True)

    forward_input = {a: input_operand}

    expected_forward = AA(result)
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward, expected_backward,
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
    exp_fwd_value.shape  = (1,1) + exp_fwd_value.shape
    exp_bkwd_value.shape = (1,) + exp_bkwd_value.shape

    # I == define cntk input variables
    a = C.input(shape=conv_input.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=True,
                name='a')

    b = C.input(shape=roi_input.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='b')

    # adding batch and sequence axis
    conv_input.shape     = (1,) + conv_input.shape
    roi_input.shape      = (1,) + roi_input.shape

    from cntk import roipooling
    input_op = roipooling(a, b, (3,3))

    forward_input = {a: conv_input, b: roi_input}
    expected_backward = {a: exp_bkwd_value}

    unittest_helper(input_op,
                    forward_input, exp_fwd_value, expected_backward,
                    device_id=device_id, precision=precision)

CONVOLUTION_TRANSPOSE_DATA = [
    ([1, 1, 3, 3], # input_size
     [1, 2, 2], # convolution size
     [[[[ 0, 0, 1, 2],
        [ 0, 5, 11, 11],
        [ 6, 23, 29, 23],
        [ 12, 32, 37, 24]]]]) # result
]
# this test handles convolution transpose, without specifying output shape
@pytest.mark.parametrize("input_size, conv_size, result", CONVOLUTION_TRANSPOSE_DATA)
def test_convolution_transpose(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution_transpose
    input_op = convolution_transpose(conv_map, a, auto_padding=[False])

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)

CONVOLUTION_TRANSPOSE_OUTPUT_DATA = [
    ([1, 1, 3, 3], # input_size
     [1, 3, 3], # convolution size
     [[[[ 0, 3, 4, 11, 8, 10],
        [ 3, 12, 11, 28, 19, 26],
        [ 12, 27, 16, 35, 20, 25],
        [ 27, 60, 35, 76, 43, 56], 
        [ 24, 51, 28, 59, 32, 40]]]]) # result
]
# this test handles convolution transpose, without specifying output shape
@pytest.mark.parametrize("input_size, conv_size, result", CONVOLUTION_TRANSPOSE_OUTPUT_DATA)
def test_convolution_transpose_with_output(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution_transpose
    input_op = convolution_transpose(conv_map, a, auto_padding=[True], strides=2, output_shape=(1,5,6))

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)
