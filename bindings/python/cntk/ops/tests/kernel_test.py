# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
from cntk.cntk_py import should_force_deterministic_algorithms

CONVOLUTION_OPERANDS = [
    ([[[5., 6.],  # (1, 2, 2) map
       [3., 4.]]],
     [[[1., 2.],  # (1, 2, 2) input operand
       [7., 8.]]],
     True),       # Use input shape with inferred dimension
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
       [11., 12.]]],
      False)      # Do not use input shape with inferred dimension
]


@pytest.mark.parametrize("convolution_map, convolution_input, use_input_shape_with_inferred_dimension", CONVOLUTION_OPERANDS)
def test_op_convolution_without_padding(convolution_map, convolution_input, use_input_shape_with_inferred_dimension, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    conv_map = AA(convolution_map, dtype=dt)
    conv_input = AA(convolution_input, dtype=dt)

    flipped_conv_map = conv_map[..., ::-1, ::-1]

    from scipy import signal
    expected_forward = AA([signal.convolve(flipped_conv_map, conv_input, mode='valid')])

    backward = AA(conv_map)

    conv_input_shape = conv_input.shape
    if use_input_shape_with_inferred_dimension:
        conv_input_shape = tuple(-1 for x in conv_input_shape)

    a = C.input_variable(shape=conv_input_shape,
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
@pytest.mark.parametrize("input_size, conv_size, result", ASYM_CONVOLUTION_DATA)
def test_asym_convolution(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
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


SPATIAL_CONVOLUTION_DATA = [
    ([1, 3, 3], # input_size
     [2, 2], # convolution size
     [[[[ 19, 25, 10],
        [ 37, 43, 16],
        [ 7, 8, 0]]]]) # result
]
# this test handles convolution with reductionRank=0.
@pytest.mark.parametrize("input_size, conv_size, result", SPATIAL_CONVOLUTION_DATA)
def test_spatial_convolution(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution
    input_op = convolution(conv_map, a, auto_padding=[True], reduction_rank=0)

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)
                    
REDUCED_OUTPUT_CONVOLUTION_DATA = [
    ([4,2,3], #input_size
     [3,2,3], # convolution size
     [[[55], [145], [235]],
      [[145],[451], [757]],
      [[235],[757], [1279]],
      [[325],[1063],[1801]]])  # result
]
# this test handles 1D/2D convolution
@pytest.mark.parametrize("input_size, conv_size, result", REDUCED_OUTPUT_CONVOLUTION_DATA)
def test_reduced_output_convolution(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution
    input_op = convolution(conv_map, a, auto_padding=[False])

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
        [ 33,   35]]]], # result
     True), # Use input shape with inferred dimension
    ([1, 1, 8, 8],
     (1, 4, 4),
     (1, 5, 5),
     [False],
     [[[[ 27 ]]]],
     False),
    ([1, 1, 6, 6],
     (1, 4, 4),
     (1, 2, 2),
     [True, False],
     [[[[ 15, 17],
        [ 27, 29],
        [ 33, 35]]]],
     True)
]
# the pooling geometry test also tests convolution geometry since they go through the same path
# in the CPU code
@pytest.mark.parametrize("input_size, pooling_window, strides, padding, result, use_input_shape_with_inferred_dimension", POOLING_GEOMETRY_DATA)
def test_op_pooling_geometry(input_size, pooling_window, strides, padding, result, use_input_shape_with_inferred_dimension, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    pool_input_shape = input_operand.shape[1:]
    if use_input_shape_with_inferred_dimension:
        pool_input_shape = tuple(-1 for x in pool_input_shape)

    a = C.input_variable(shape=pool_input_shape,
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

    a = C.sequence.input_variable(shape=input_operand.shape[2:],
                         dtype=sanitize_dtype_cntk(precision),
                         needs_gradient=True,
                         name='a')

    backward = (1 / np.prod(pooling_window)) * np.ones_like(input_operand)

    from cntk import pooling
    # The test example data above do not use padding.
    # Expected backward result is also computed based on no padding.
    # So here auto_padding is explicitly set to False.
    # auto_padding = True is tested in a separate test case below.
    input_op = pooling(a, AVG_POOLING, pooling_window, strides, auto_padding=[False])

    forward_input = {a: input_operand}

    expected_forward = AA([result])
    expected_backward = {a: backward}

    unittest_helper(input_op, forward_input, expected_forward,
                expected_backward, device_id=device_id, precision=precision)

AVG_POOLING_AUTOPAD_DATA = [
    ([1, 1, 1, 6, 6],
     (1, 5, 3),
     (1, 1, 1),
     [[[[ 7.5,  8.,   9.,  10.,  11.,  11.5,],
        [10.5, 11.,  12.,  13.,  14.,  14.5,],
        [13.5, 14.,  15.,  16.,  17.,  17.5,],
        [19.5, 20.,  21.,  22.,  23.,  23.5,],
        [22.5, 23.,  24.,  25.,  26.,  26.5,],
        [25.5, 26.,  27.,  28.,  29.,  29.5,]]]],
     [[[[[0.6527778,  0.9138889,  0.7833333,  0.7833333,  0.91388893, 0.65277785],
         [0.8194445,  1.1472223,  0.9833333,  0.9833333,  1.1472223,  0.81944454],
         [1.0277778,  1.438889,   1.2333333,  1.2333333,  1.438889,   1.0277779 ],
         [1.0277778,  1.438889,   1.2333333,  1.2333333,  1.4388889,  1.0277778 ],
         [0.8194445,  1.1472223,  0.9833333,  0.9833333,  1.1472222,  0.8194445 ],
         [0.6527778,  0.91388893, 0.78333336, 0.78333336, 0.91388893, 0.6527778 ]]]]]),
    ([1, 1, 1, 6, 6],
     (1, 5, 3),
     (1, 2, 2),
     [[[[ 7.5,  9.,  11. ],
        [13.5, 15.,  17. ],
        [22.5, 24.,  26. ]]]],
     [[[[[0.26666668, 0.44444448, 0.17777778, 0.35555556, 0.17777778, 0.17777778,],
         [0.26666668, 0.44444448, 0.17777778, 0.35555556, 0.17777778, 0.17777778,],
         [0.39166668, 0.6527778,  0.2611111,  0.5222222,  0.2611111,  0.2611111, ],
         [0.225,      0.375,      0.15,       0.3,        0.15,       0.15,      ],
         [0.225,      0.375,      0.15,       0.3,        0.15,       0.15,      ],
         [0.125,      0.20833334, 0.08333334, 0.16666667, 0.08333334, 0.08333334,]]]]])
]
@pytest.mark.parametrize("input_size, pooling_window, strides, result, backward_result", AVG_POOLING_AUTOPAD_DATA)
def test_op_avg_pooling_auto_padding(input_size, pooling_window, strides, result, backward_result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.sequence.input_variable(shape=input_operand.shape[2:],
                         dtype=sanitize_dtype_cntk(precision),
                         needs_gradient=True,
                         name='a')

    from cntk import pooling
    input_op = pooling(a, AVG_POOLING, pooling_window, strides, auto_padding=[True])

    forward_input = {a: input_operand}

    expected_forward = AA([result])
    expected_backward = {a: backward_result}

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
        [ 59.,  61.,  63.,  64.]]]]),

    ([1, 1, 1, 6, 6],
     (5, 3),
     (1, 1),
     [True],
     [[[[14., 15., 16., 17., 18., 18.,],
        [20., 21., 22., 23., 24., 24.,],
        [26., 27., 28., 29., 30., 30.,],
        [32., 33., 34., 35., 36., 36.,],
        [32., 33., 34., 35., 36., 36.,],
        [32., 33., 34., 35., 36., 36.,]]]])
]


@pytest.mark.parametrize("input_size, pooling_window, strides, autopad, result", MAX_POOLING_DATA)
def test_op_max_pooling(input_size, pooling_window, strides, autopad, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.sequence.input_variable(shape=input_operand.shape[2:],
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

    a = C.sequence.input_variable(shape=input_operand.shape[2:],
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

    # backward are not all ones when there is padding.
    expected_forward = (backward > 0).astype(np.float32) * input_operand
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

    a = C.input_variable(shape=input_operand.shape[1:], dtype=sanitize_dtype_cntk(precision), needs_gradient=True, name='a')

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

    a = C.input_variable(shape=input_operand.shape[1:], dtype=sanitize_dtype_cntk(precision), needs_gradient=True, name='a')

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
     [[1, 1, 2, 2]],       # (4) input roi (x1, y1, x2, y2), where (x1, y1) is top left coordinate and (x2, y2) bottom right coordinate.
     [[[5., 6., 6.],       # (1, 3, 3) expected forward output
       [8., 9., 9.],
       [8., 9., 9.]]],
     [[[0., 0., 0.],       # (1, 3, 3) expected backward output (gradient input is all 1s)
       [0., 1., 2.],
       [0., 2., 4.]]])
]

@pytest.mark.parametrize("input_map, input_rois, expected_fwd, expected_bkwd", ROIPOOLING_OPERANDS)
def test_op_maxroipooling(input_map, input_rois, expected_fwd, expected_bkwd, device_id, precision):
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
    a = C.input_variable(shape=conv_input.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=True,
                name='a')

    b = C.input_variable(shape=roi_input.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='b')

    # adding batch and sequence axis
    conv_input.shape     = (1,) + conv_input.shape
    roi_input.shape      = (1,) + roi_input.shape

    from cntk import roipooling
    input_op = roipooling(a, b, C.MAX_POOLING, (3,3), 1.)

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

    a = C.input_variable(shape=input_operand.shape[1:],
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

    a = C.input_variable(shape=input_operand.shape[1:],
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


SPATIAL_CONVOLUTION_TRANSPOSE_DATA = [
    ([1, 3, 3], # input_size
     [2, 2], # convolution size
     [[[[ 0, 0, 1, 2],
        [ 0, 5, 11, 11],
        [ 6, 23, 29, 23],
        [ 12, 32, 37, 24]]]]) # result
]
# this test handles convolution transpose, without specifying output shape and with reduction_rank=0
@pytest.mark.parametrize("input_size, conv_size, result", SPATIAL_CONVOLUTION_TRANSPOSE_DATA)
def test_spatial_convolution_transpose(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution_transpose
    input_op = convolution_transpose(conv_map, a, auto_padding=[False], reduction_rank=0)

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)

SPATIAL_CONVOLUTION_TRANSPOSE_OUTPUT_DATA = [
    ([1, 3, 3], # input_size
     [3, 3], # convolution size
     [[[[ 0, 3, 4, 11, 8, 10],
        [ 3, 12, 11, 28, 19, 26],
        [ 12, 27, 16, 35, 20, 25],
        [ 27, 60, 35, 76, 43, 56], 
        [ 24, 51, 28, 59, 32, 40]]]]) # result
]
# this test handles convolution transpose, without specifying output shape and with reduction_rank=0
@pytest.mark.parametrize("input_size, conv_size, result", SPATIAL_CONVOLUTION_TRANSPOSE_OUTPUT_DATA)
def test_spatial_convolution_transpose_with_output(input_size, conv_size, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution_transpose
    input_op = convolution_transpose(conv_map, a, auto_padding=[True], strides=2, output_shape=(1,5,6), reduction_rank=0)

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)

def test_conv_incorrect_shapes():
    input = C.input_variable(())
    with pytest.raises(ValueError):
        h = C.layers.Convolution(filter_shape=(5,5), num_filters=8, strides=(1,1), pad=True)(input)
    with pytest.raises(ValueError):
        h = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(input)

    input = C.input_variable(28)
    with pytest.raises(ValueError):
        h = C.layers.Convolution(filter_shape=(5,5), num_filters=8, strides=(1,1), pad=True)(input)
    with pytest.raises(ValueError):
        h = C.layers.MaxPooling(filter_shape=(2,2), strides=(2,2))(input)

def test_conv_cudnn_batch_size_change(device_id):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')

    np.random.seed(0)
    input_shape = (1, 16, 100)
    input1 = C.sequence.input_variable(input_shape, needs_gradient=True, sequence_axis=C.Axis.new_unique_dynamic_axis('c'))
    input2 = C.sequence.input_variable(input_shape, needs_gradient=True, sequence_axis=C.Axis.new_unique_dynamic_axis('q'))
    conv = C.layers.Convolution2D((5,8), 100, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0)
    output = C.reduce_sum(conv(input1), axis=C.Axis.all_axes()) + C.reduce_sum(conv(input2), axis=C.Axis.all_axes())
    num_batches = 100 # change to greater value for a more thorough test
    batch_size = 1
    max_seq_len = [100, 10]
    for batch in range(num_batches):
        seq_lens = [[int(x*msl+1) for x in np.random.random((batch_size))] for msl in max_seq_len]
        output.grad({input1:[np.random.random((sl,) + input_shape).astype(np.float32) for sl in seq_lens[0]],
                     input2:[np.random.random((sl,) + input_shape).astype(np.float32) for sl in seq_lens[1]]})

FREE_STATIC_AXES_CONVOLUTION_DATA = [
    # 2D convolution with single free static axis.
    ([3, 101, 151], # warmup_input_size: Defines the input size used for first run with free static axes. 3- and 4-element vector for 2D and 3D convolution, respectively. 
     [200],         # free_dimension_increment: Increments to the input size for the second/actual/test run. Length defines the number of free static axes.
     [5, 5],        # filter_size: kernel size for convolution. Length defines 2D or 3D convolution.
     32            # num_output_channels
     ),
    # 2D convolution with two free static axes.
    ([3, 51, 101], 
     [30, 20], 
     [3, 3], 
     64
     ),
    # 3D convolution with three free static axes.
    ([3, 51, 101, 71],
     [10, 20, 40],
     [3, 3, 3],
     8
     ),
    # 3D convolution with two free static axes.
    ([5, 51, 61, 91],
     [6, 8],
     [3, 3, 3],
     8
     ),
    # 3D convolution with single free static axis.
    ([2, 101, 121, 151],
     [10],
     [3, 3, 3],
     4
     )
]
# This test point exercises 2D and 3D convolution with single and multiple free static axes, and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("warmup_input_size, free_dimension_increment, filter_size, num_output_channels", FREE_STATIC_AXES_CONVOLUTION_DATA)
def test_conv_free_static_axes(warmup_input_size, free_dimension_increment, filter_size, num_output_channels, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    conv_size = tuple([num_output_channels, warmup_input_size[0]]+filter_size)
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    reference_input_size = tuple(warmup_input_size[:-len(free_dimension_increment)] +
                           [x+y for x,y in zip(warmup_input_size[-len(free_dimension_increment):], free_dimension_increment)])

    a_ref = C.input_variable(shape=reference_input_size,
                dtype=dt,
                needs_gradient=False,
                name='a_ref')
    a_test = C.input_variable(shape=tuple(warmup_input_size[:-len(free_dimension_increment)] + [C.FreeDimension]*len(free_dimension_increment)),
                dtype=dt,
                needs_gradient=False,
                name='a_test')

    from cntk import convolution

    conv_op_without_free_dim = convolution(conv_map, a_ref, auto_padding=[False] + [True]*len(filter_size))
    conv_op_with_free_dim = convolution(conv_map, a_test, auto_padding=[False] + [True]*len(filter_size))

    input_img_ref = np.ones(reference_input_size, dtype=dt)
    output_ref = conv_op_without_free_dim.eval({a_ref: input_img_ref}, device=dev)

    input_img_warmup = np.ones(warmup_input_size, dtype=dt)
    _ = conv_op_with_free_dim.eval({a_test: input_img_warmup}, device=dev)
        
    output_test = conv_op_with_free_dim.eval({a_test: input_img_ref}, device=dev)

    assert np.allclose(output_test, output_ref, atol = 1e-4)

FREE_STATIC_AXES_WITH_DYNAMIC_AXIS_CONVOLUTION_DATA = [    
    # 2D convolution with two free static axes and one batch (dynamic) axis.
    ([3, 31, 51], # warmup_input_size: Defines the input size used for first run with free static axes. 3- and 4-element vector for 2D and 3D convolution, respectively.
     [10, 12],    # free_dimension_increment: Increments to the input size for the second/actual/test run. Length defines the number of free static axes.
     [3, 3],      # filter_size: kernel size for convolution. Length defines 2D or 3D convolution.
     16,          # num_output_channels
     [2, 10]      # Half-open range for random selection of of batch-sizes (for reference and warmup)
     ),        
]
# This test point exercises convolution/pooling/unpooling with multiple free static axes and batch (dynamic) axis), and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("warmup_input_size, free_dimension_increment, filter_size, num_output_channels, batch_size_range", FREE_STATIC_AXES_WITH_DYNAMIC_AXIS_CONVOLUTION_DATA)
def test_conv_pooling_free_static_and_dynamic_axes(warmup_input_size, free_dimension_increment, filter_size, num_output_channels, batch_size_range, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    np.random.seed(0)

    conv_size = tuple([num_output_channels, warmup_input_size[0]]+filter_size)
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    warmup_batchsize = np.random.randint(batch_size_range[0],batch_size_range[1])
    ref_batchsize = np.random.randint(batch_size_range[0],batch_size_range[1])

    reference_input_size = tuple(warmup_input_size[:-len(free_dimension_increment)] +
                           [x+y for x,y in zip(warmup_input_size[-len(free_dimension_increment):], free_dimension_increment)])

    a_ref = C.sequence.input_variable(shape=reference_input_size,
                dtype=dt,
                needs_gradient=False,
                sequence_axis=C.Axis.new_unique_dynamic_axis('c'))
    a_test = C.sequence.input_variable(shape=tuple(warmup_input_size[:-len(free_dimension_increment)] + [C.FreeDimension]*len(free_dimension_increment)),
                dtype=dt,
                needs_gradient=False,
                sequence_axis=C.Axis.new_unique_dynamic_axis('c'))

    from cntk import convolution, pooling, unpooling

    def pooling_unpooling(x):
        y = pooling(x, C.AVG_POOLING, (2,2), (2,2), auto_padding=[True])
        return unpooling(y, x, C.MAX_UNPOOLING, (2,2), (2,2), auto_padding=[True])

    conv_ops = [ [convolution(conv_map, a_ref, auto_padding=[False] + [True]*len(filter_size)),
                  convolution(conv_map, a_test, auto_padding=[False] + [True]*len(filter_size))],
                 [pooling_unpooling(a_ref),
                  pooling_unpooling(a_test)] ]

    for op_pair in conv_ops:
        conv_op_without_free_dim, conv_op_with_free_dim = op_pair
        input_img_ref = np.random.random((ref_batchsize,) + reference_input_size).astype(dt)
        output_ref = conv_op_without_free_dim.eval({a_ref: input_img_ref}, device=dev)

        input_img_warmup = np.random.random((warmup_batchsize,) + tuple(warmup_input_size)).astype(dt)
        _ = conv_op_with_free_dim.eval({a_test: input_img_warmup}, device=dev)

        output_test = conv_op_with_free_dim.eval({a_test: input_img_ref}, device=dev)

        assert np.allclose(output_test, output_ref, atol = 1e-4)

DILATED_CONVOLUTION_DATA = [
    # Dilation without passing.
    ([1, 1, 5, 5], # input_size
     [1, 3, 3],    # convolution size
     [[[[624]]]],  # result
     False),       # Padding 
     # Dilation with padding
    ([1, 1, 5, 5],                              # input_size
     [1, 3, 3],                                 # convolution size
     [[[[176,  200,  284,  172,  192],
        [296,  320,  449,  272,  292],
        [420,  447,  624,  375,  396],
        [164,  176,  233,  128,  136],
        [224,  236,  308,  168,  176]]]],       # result
     True)                                      # Padding 
]
@pytest.mark.parametrize("input_size, conv_size, result, padding", DILATED_CONVOLUTION_DATA)
def test_convolution_dilated(input_size, conv_size, result, padding, device_id, precision):
    if device_id == -1:
        pytest.skip('Test only runs on GPU')

    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # fill input operand with a sequence 0,1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(total_size, dtype=dt)
    input_operand = x.reshape(input_size)

    a = C.input_variable(shape=input_operand.shape[1:],
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')

    # do the same for convolution kernel
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt)
    conv_map = constant(value=y.reshape(conv_size), device=dev)

    from cntk import convolution
    input_op = convolution(conv_map, a, auto_padding=[padding], dilation=2)

    forward_input = {a: input_operand}
    expected_forward = AA(result)

    unittest_helper(input_op, forward_input, expected_forward,
                    None, device_id=device_id, precision=precision)

FREE_STATIC_AXES_SEQUENCE_UNPACK_CONVOLUTION_DATA = [    
    # Convolution with free static axes using sequence unpack.
    (6,         # num_features: Defines the input size used for first run with free static axes. 3- and 4-element vector for 2D and 3D convolution, respectively.
     4,         # sequence_len: Number of sequences.
     [3, 3],    # filter_size: kernel size for convolution. Length defines 2D or 3D convolution.
     8,        # num_output_channels
     2          # batch_size
     )
]
# This test point exercises convolution with free static axes produced by sequence.unpack, and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("num_features, sequence_len, filter_size, num_output_channels, batch_size", FREE_STATIC_AXES_SEQUENCE_UNPACK_CONVOLUTION_DATA)
def test_conv_free_static_with_sequence_unpack(num_features, sequence_len, filter_size, num_output_channels, batch_size, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    x_ref = C.input_variable((1, sequence_len, num_features), dtype=dt)
    conv_map_ref = C.constant(np.random.randn(num_output_channels, 1, filter_size[0], filter_size[1]).astype(dt), device=dev)
    w2_ref = C.convolution(conv_map_ref, x_ref, auto_padding=[False])
    x0_ref = np.arange(batch_size*1*sequence_len*num_features).astype(dt).reshape(batch_size, 1, sequence_len, num_features)
    output_ref = w2_ref.eval({x_ref: x0_ref}, device=dev)


    x_test = C.sequence.input_variable(num_features, dtype=dt)
    y_test, mask_test = C.sequence.unpack(x_test, 0).outputs
    z_test = C.reshape(y_test, (1, ), 0, 0)
    w2_test = C.convolution(conv_map_ref, z_test, auto_padding=[False])
    output_test = w2_test.eval({x_test: np.squeeze(x0_ref)}, device=dev)
    
    assert np.allclose(output_test, output_ref, atol=1e-4)

GROUP_CONVOLUTION_DATA = [
    # 2D Convolution.
    (4,          # groups
     112,        # num_output_channels
     24,         # num_input_channels
     [30, 40],   # input_tensor_size (not including channels)
     [3, 3],     # filter_size: kernel size for convolution. Length defines 2D or 3D convolution.
     6,          # kernel_channels: kC, number of input channels in kernel
     2           # batch_size
     ),
    # 3D Convolution.
    (2,              # groups
     10,             # num_output_channels
     6,              # num_input_channels
     [15, 25, 30],   # input_tensor_size (not including channels)
     [3, 5, 7],      # filter_size: kernel size for convolution. Length defines 2D or 3D convolution.
     3,              # kernel_channels: kC, number of input channels in kernel
     2               # batch_size
     )
]
# This test point exercises group convolution, and tests against grouping simulated explicitly using convolution without grouping.
@pytest.mark.parametrize("groups, num_output_channels, num_input_channels, input_tensor_size, filter_size, kernel_channels, batch_size", GROUP_CONVOLUTION_DATA)
def test_group_conv(groups, num_output_channels, num_input_channels, input_tensor_size, filter_size, kernel_channels, batch_size, device_id, precision):
    if device_id == -1 and len(input_tensor_size) > 2:
        pytest.skip('3D or higher dimensions not supported for group convolution on CPU.')
    if device_id == 0 and should_force_deterministic_algorithms():
        pytest.skip('Deterministic algorithms not supported on GPU for group convolution.')

    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # Generate result from CNTK API
    conv_size = tuple([num_output_channels, kernel_channels]+filter_size)
    total_size = np.prod(conv_size)
    y = np.arange(total_size, dtype=dt).reshape(conv_size)
    conv_map = C.parameter(init=y, device=dev)

    input_size = (num_input_channels, ) + tuple(input_tensor_size)
    x_test = C.input_variable(input_size, needs_gradient=True, dtype=dt)
    data = np.random.random((batch_size,) + input_size).astype(dt)

    conv_op = C.convolution(conv_map, x_test, auto_padding=[False] + [True]*len(filter_size), groups = groups)

    df_test, fv_test = conv_op.forward({x_test:data}, [conv_op.output], set([conv_op.output]), device=dev)
    output_test = list(fv_test.values())[0]
    grad_data = np.random.random(size=output_test.shape)
    grad_test = conv_op.backward(df_test, {conv_op.output: grad_data}, set([x_test]))
    output_grad_test = list(grad_test.values())[0]

    # Generate reference result. The code below simulates (actually is just another implementation in Python)
    # group convolution using multiple standard convolutions (i.e. groups = 1), to create the reference
    # output for testing the CNTK implementation against.     
    num_out_channels_per_group = int(num_output_channels / groups)
    num_in_channels_per_group = int(num_input_channels / groups)
    sub_kernels_init = [y[i * num_out_channels_per_group:(i+1) * num_out_channels_per_group, ...] for i in range(0, groups)]
    sub_kernels = [C.ops.parameter(init=np.ascontiguousarray(sub_kernels_init[i]), device=dev)
                          for i in range(0, groups)]                          

    x_ref = C.input_variable(input_size, needs_gradient=True, dtype=dt)                                             
    sub_data = [C.ops.slice(x_ref, axis=0, begin_index=i * num_in_channels_per_group,
                             end_index=(i + 1) * num_in_channels_per_group) for i in range(0, groups)]
    conv_ops_per_group = [C.ops.convolution(group_kernel, data_for_groups, auto_padding=[False] + [True]*len(filter_size)) 
                 for group_kernel, data_for_groups in zip(sub_kernels, sub_data)]
    group_conv = C.ops.splice(*conv_ops_per_group, axis=0)

    df_ref, fv_ref = group_conv.forward({x_ref:data}, [group_conv.output], set([group_conv.output]), device=dev)
    output_ref = list(fv_ref.values())[0]
    grad_ref = group_conv.backward(df_ref, {group_conv.output: grad_data}, set([x_ref]))
    output_grad_ref = list(grad_ref.values())[0]

    assert np.allclose(output_test, output_ref, atol=1e-4)
    assert np.allclose(output_grad_test, output_grad_ref, atol=1e-4)

def test_group_conv_shape(device_id):
    x = C.input_variable((16, 64, 64))
    param = C.parameter((16, 1, 3, 3))
    y_pad_channel = C.convolution(param, x, groups=16)
    y_pad_channel_2 = C.convolution(param, x, groups=16, auto_padding=[True, True, True])
    y_not_pad_channel = C.convolution(param, x, groups=16, auto_padding=[False, True, True])
    
    # Though in most cases unintended, padding in channel axis is expected and supported behavior when auto_padding is not specified or set to [True, ..]. 
    assert np.allclose(y_pad_channel.shape, (256, 64, 64))
    assert np.allclose(y_pad_channel_2.shape, (256, 64, 64))
    # Explicit specification is required if we don't want padding on channel
    assert np.allclose(y_not_pad_channel.shape, (16, 64, 64))


FREE_STATIC_AXES_MAX_POOLING_DATA = [
    ((1, 4, 6, 6), # warmup_input_size: Defines the input size used for first run with free static axes.
     (1, 4, 6, 9), # second_input_size: Defines the input size used for second run with free static axes.
     (2, 2),       # pooling_window: Dimensions of the pooling window.
     (2, 2)        # strides
     )
]
# This test point exercises maxpooling with free static axes twice (first for warmup, second for actual test), 
# and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("warmup_input_size, second_input_size, pooling_window, strides", FREE_STATIC_AXES_MAX_POOLING_DATA)
def test_max_pooling_free_static_axes(warmup_input_size, second_input_size, pooling_window, strides, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # Exercise operation twice - once with warmup input, second time to get the test output.
    x = C.input_variable((warmup_input_size[0:2]+ tuple([C.FreeDimension]*(len(warmup_input_size)-2))))
    y = C.pooling(x, C.MAX_POOLING, pooling_window, strides)
    
    x_data_warmup = np.arange(np.prod(warmup_input_size), dtype=dt)
    x_data_warmup = x_data_warmup.reshape(warmup_input_size)
    output_warmup = y.eval({x:x_data_warmup}, device=dev)
    
    x_data_test = np.arange(np.prod(second_input_size), dtype=dt)
    x_data_test = x_data_test.reshape(second_input_size)
    output_test = y.eval({x:x_data_test}, device=dev)
    
    # Generate reference output using fixed axes.
    x_ref = C.input_variable(second_input_size)
    y_ref = C.pooling(x_ref, C.MAX_POOLING, pooling_window, strides)
    output_ref = y_ref.eval({x_ref:x_data_test}, device=dev)

    assert np.allclose(output_test, output_ref, atol=1e-4)

FREE_STATIC_AXES_AVG_POOLING_DATA = [
    ((1, 4, 6, 6), # warmup_input_size: Defines the input size used for first run with free static axes.
     (1, 4, 6, 9), # second_input_size: Defines the input size used for second run with free static axes.
     (2, 2),       # pooling_window: Dimensions of the pooling window.
     (2, 2)        # strides
     )
]
# This test point exercises average pooling with free static axes twice (first for warmup, second for actual test), 
# and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("warmup_input_size, second_input_size, pooling_window, strides", FREE_STATIC_AXES_AVG_POOLING_DATA)
def test_avg_pooling_free_static_axes(warmup_input_size, second_input_size, pooling_window, strides, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # Exercise operation twice - once with warmup input, second time to get the test output.
    x = C.input_variable((warmup_input_size[0:2]+ tuple([C.FreeDimension]*(len(warmup_input_size)-2))))
    y = C.pooling(x, C.AVG_POOLING, pooling_window, strides)
    
    x_data_warmup = np.arange(np.prod(warmup_input_size), dtype=dt)
    x_data_warmup = x_data_warmup.reshape(warmup_input_size)
    output_warmup = y.eval({x:x_data_warmup}, device=dev)
    
    x_data_test = np.arange(np.prod(second_input_size), dtype=dt)
    x_data_test = x_data_test.reshape(second_input_size)
    output_test = y.eval({x:x_data_test}, device=dev)
    
    # Generate reference output using fixed axes.
    x_ref = C.input_variable(second_input_size)
    y_ref = C.pooling(x_ref, C.AVG_POOLING, pooling_window, strides)
    output_ref = y_ref.eval({x_ref:x_data_test}, device=dev)

    assert np.allclose(output_test, output_ref, atol=1e-4)

FREE_STATIC_AXES_MAX_UNPOOLING_DATA = [
    ((1, 4, 6, 6), # warmup_input_size: Defines the input size used for first run with free static axes.
     (1, 4, 6, 9), # second_input_size: Defines the input size used for second run with free static axes.
     (2, 2),       # pooling_window: Dimensions of the pooling window.
     (2, 2)        # strides
     )
]
# This test point exercises max unpooling with free static axes twice (first for warmup, second for actual test), 
# and ensures that the result is the same as with fixed axes.
@pytest.mark.parametrize("warmup_input_size, second_input_size, pooling_window, strides", FREE_STATIC_AXES_MAX_UNPOOLING_DATA)
def test_max_unpooling_free_static_axes(warmup_input_size, second_input_size, pooling_window, strides, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    # Exercise operation twice - once with warmup input, second time to get the test output.
    x = C.input_variable((warmup_input_size[0:2]+ tuple([C.FreeDimension]*(len(warmup_input_size)-2))))
    y = C.pooling(x, C.MAX_POOLING, pooling_window, strides)
    z = C.unpooling(y, x, C.MAX_UNPOOLING, pooling_window, strides)
    
    x_data_warmup = np.arange(np.prod(warmup_input_size), dtype=dt)
    x_data_warmup = x_data_warmup.reshape(warmup_input_size)
    output_warmup = z.eval({x:x_data_warmup}, device=dev)
    
    x_data_test = np.arange(np.prod(second_input_size), dtype=dt)
    x_data_test = x_data_test.reshape(second_input_size)
    output_test = z.eval({x:x_data_test}, device=dev)
    
    # Generate reference output using fixed axes.
    x_ref = C.input_variable(second_input_size)
    y_ref = C.pooling(x_ref, C.MAX_POOLING, pooling_window, strides)
    z_ref = C.unpooling(y_ref, x_ref, C.MAX_UNPOOLING, pooling_window, strides)
    output_ref = z_ref.eval({x_ref:x_data_test}, device=dev)

    assert np.allclose(output_test, output_ref, atol=1e-4)