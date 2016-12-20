# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for as_block operation, only forward pass is tested
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, cntk_device
import cntk as C
from cntk.axis import Axis
from ...utils import sanitize_dtype_cntk
from .. import constant

AS_BLOCK_TEST_CASES = [
    #(input_shape, output_shape, expected_output_shape)
    ((2, 3),    (3, 2), (3, 2)),
    ((2, 3),    (6, 1), (6, 1)),
    ((6, 1),    (2, 3), (2, 3)),
    ((2, 3, 5), (5, 6), (5, 6)),
    ((2, 3, 5), (C.InferredDimension, 6), (5, 6)),
    ((2, 3, 5), (5, C.InferredDimension), (5, 6)),
]

@pytest.mark.parametrize("input_shape, output_shape, expected_output_shape", AS_BLOCK_TEST_CASES)
def test_op_as_block(input_shape, output_shape, expected_output_shape, device_id, precision):
    # We test using reshape as the operation that is encapsulated in a block

    dev = cntk_device(device_id)
    from ...utils import sanitize_dtype_cntk
    from .. import reshape, element_times, as_block

    num_tensor_elements = np.multiply.reduce(input_shape)
    input_tensor = np.arange(
        num_tensor_elements, dtype=PRECISION_TO_TYPE[precision]).reshape(input_shape)
    input_reshaped = input_tensor.reshape(expected_output_shape)

    a_placeholder = C.placeholder_variable();
    a_reshaped = reshape(a_placeholder, output_shape)

    const_input_reshaped = constant(input_reshaped, device=dev)
    block_composite = element_times(a_reshaped, const_input_reshaped)
    
    a = I(shape=input_tensor.shape,
          dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    input_op = as_block(block_composite, {a_placeholder : a}, 'reshape_test_op')

    expected_forward = [[input_reshaped**2]]
    expected_backward = {a: input_tensor}

    # create batch
    input_tensor.shape = (1, 1) + input_tensor.shape

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)
