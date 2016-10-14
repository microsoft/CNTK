# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for kernel operations, tested for the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE, constant
from ...utils import sanitize_dtype_cntk

CONVOLUTION_OPERANDS = [
    ([[[5., 6.], # (1, 2, 2) map
       [3., 4.]]],
     [[[1., 2.], # (1, 2, 2) input operand
       [7., 8.]]]),
    ([[[1., 2.], # (3, 2, 2) map
       [3., 4.]],
      [[1., 2.],
       [3., 4.]],
      [[1., 2.],
       [3., 4.]]],
     [[[1., 2.], # (3, 2, 2) input operand
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

    conv_input.shape = (1,1) + conv_input.shape # adding batch and channel axis
    conv_map.shape = (1,1) + conv_map.shape

    flipped_conv_map = conv_map[...,::-1,::-1]

    from scipy import signal
    expected_forward = AA([[signal.convolve(flipped_conv_map, conv_input, mode='valid')]])

    backward = AA([[conv_map]])

    a = I(shape=conv_input.shape,
        data_type=sanitize_dtype_cntk(precision),
        needs_gradient=True,
        name='a')

    constant_map = constant(value=conv_map)

    from cntk import convolution
    input_op = convolution(constant_map, a, auto_padding=[False])

    conv_input.shape = (1, 1) + conv_input.shape
    forward_input = {a: conv_input}
    expected_backward = {a: backward}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)
