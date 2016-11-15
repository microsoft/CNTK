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
from ...utils import sanitize_dtype_cntk

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

    conv_map = AA(convolution_map, dtype=dt)
    conv_input = AA(convolution_input, dtype=dt)

    # adding batch and channel axis
    conv_input.shape = (1, 1) + conv_input.shape
    conv_map.shape = (1, 1) + conv_map.shape

    flipped_conv_map = conv_map[..., ::-1, ::-1]

    from scipy import signal
    expected_forward = AA(
        [[signal.convolve(flipped_conv_map, conv_input, mode='valid')]])

    backward = AA(conv_map)

    a = I(shape=conv_input.shape,
        dtype=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    constant_map = constant(value=conv_map)

    from cntk import convolution
    input_op = convolution(constant_map, a, auto_padding=[False])

    forward_input = {a: conv_input}
    expected_backward = {a: [[backward]]}

    unittest_helper(input_op, forward_input, expected_forward,
                    expected_backward, device_id=device_id, precision=precision)

AVG_POOLING_DATA = [
    ([1, 2, 2, 4, 3],  # input_size
     (1, 2, 2, 2, 1),  # pooling_window
     (1, 2, 2, 2, 1),  # strides
     [[[[[20.5,  21.5,  22.5],
         [26.5,  27.5,  28.5]]]]]),  # result
    ([1, 2, 4, 4, 4],
     (1, 2, 2, 2, 2),
     (1, 2, 2, 2, 2),
     [[[[[43.5,  45.5],
         [51.5,  53.5]],
        [[75.5,  77.5],
         [83.5,  85.5]]]]]),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, result", AVG_POOLING_DATA)
def test_op_avg_pooling(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    expected_forward = AA([[result]])

    backward = (1 / np.prod(pooling_window)) * np.ones_like(input_operand)

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import pooling

    _test_unary_op(precision, device_id, pooling, input_operand, expected_forward, expected_backward, {
                   'pooling_type': AVG_POOLING, 'pooling_window_shape': pooling_window, 'strides': strides, 'auto_padding': [True]})

MAX_POOLING_DATA = [
    ([1, 2, 2, 4, 3],  # input_size
     (1, 2, 2, 2, 1),  # pooling_window
     (1, 2, 2, 2, 1),  # strides
     [[[[[40.,  41.,  42.],
         [46.,  47.,  48.]]]]]),  # result
    ([1, 2, 4, 4, 4],
     (1, 2, 2, 2, 2),
     (1, 2, 2, 2, 2),
     [[[[[86.,   88.],
         [94.,   96.]],
        [[118.,  120.],
         [126.,  128.]]]]]),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, result", MAX_POOLING_DATA)
def test_op_max_pooling(input_size, pooling_window, strides, result, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    result_array = np.asarray(result, dtype=dt)
    max_elements = result_array.reshape(result_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(input_operand)
    for element in max_elements:
        backward += np.asarray(input_operand == element)

    expected_forward = AA([[result]])

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import pooling
    _test_unary_op(precision, device_id, pooling, input_operand, expected_forward, expected_backward, {
                   'pooling_type': MAX_POOLING, 'pooling_window_shape': pooling_window, 'strides': strides})

CONV_WITH_PADDING_OPERANDS = [
    ([[[1., 2., 3.],  # (3, 2, 3) map
       [3., 4., 3.]],
      [[1., 2., 3.],
       [3., 4., 3.]],
      [[1., 2., 3.],
       [3., 4., 3.]]],
     [[[1., 2., 3.],  # (3, 2, 3) input operand
       [3., 4., 6.]],
      [[5., 6., 7.],
       [7., 8., 9.]],
      [[9., 10., 11.],
       [11., 12., 12.]]],
     [[[[[112.,  175.,  124.],  # (3, 2, 3) expected_forward
         [56.,   79.,   42.]],
        [[240.,  354.,  240.],
         [114.,  150.,   78.]],
        [[208.,  297.,  196.],
         [96.,  121.,   62.]]]]],
     [[[[[6.,  12.,  10.],  # (3, 2, 3) expected_backward
         [20.,  32.,  24.]],
        [[9.,  18.,  15.],
         [30.,  48.,  36.]],
        [[6.,  12.,  10.],
         [20.,  32.,  24.]]]]]
     )
]


@pytest.mark.parametrize("convolution_map, convolution_input, forward, backward", CONV_WITH_PADDING_OPERANDS)
def test_op_convolution_with_padding(convolution_map, convolution_input, forward, backward, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    conv_map = AA(convolution_map, dtype=dt)
    conv_input = AA(convolution_input, dtype=dt)

    # adding batch and channel axis
    conv_input.shape = (1, 1) + conv_input.shape
    conv_map.shape = (1, 1) + conv_map.shape

    expected_forward = AA([[forward]], dtype=dt)

    a = I(shape=conv_input.shape,
          dtype=sanitize_dtype_cntk(precision),
          needs_gradient=True,
          name='a')

    constant_map = constant(value=conv_map)

    from cntk import convolution
    input_op = convolution(constant_map, a, auto_padding=[True])

    forward_input = {a: conv_input}
    expected_backward = {a: AA([[backward]], dtype=dt)}

    unittest_helper(input_op, forward_input, expected_forward,
                    expected_backward, device_id=device_id, precision=precision)

AVG_POOLING_DATA_WITH_STRIDES = [
    ([1, 2, 2, 4, 3],  # input size
     (1, 2, 2, 2, 1),  # pooling window size
     (1, 2, 2, 1, 1),  # strides
     [[[[[20.5,  21.5,  22.5],  # expected forward
         [23.5,  24.5,  25.5],
         [26.5,  27.5,  28.5]]]]],
     np.tile([[0.125,  0.125,  0.125],  # expected backward
              [0.25,  0.25,  0.25],
              [0.25,  0.25,  0.25],
              [0.125,  0.125,  0.125]], (1, 2, 2, 1, 1))),
]


@pytest.mark.parametrize("input_size, pooling_window, strides, forward, backward", AVG_POOLING_DATA_WITH_STRIDES)
def test_op_avg_pooling_with_different_strides(input_size, pooling_window, strides, forward, backward, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    # fill input operand with a sequence 1,2,3,... til total size and then
    # resize to input_size
    total_size = np.prod(input_size)
    x = np.arange(1, total_size + 1, 1, dtype=dt)
    input_operand = x.reshape(input_size)

    expected_forward = AA([[forward]], dtype=dt)

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import pooling
    _test_unary_op(precision, device_id, pooling, input_operand, expected_forward, expected_backward, {
                   'pooling_type': AVG_POOLING, 'pooling_window_shape': pooling_window, 'strides': strides})
