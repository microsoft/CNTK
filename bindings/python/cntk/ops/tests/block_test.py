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
    block_composite = element_times(a_reshaped, const_input_reshaped, name='element_times_inside_block')
    
    a = I(shape=input_tensor.shape,
          dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
          needs_gradient=True,
          name='a')

    input_op = as_block(block_composite, [(a_placeholder, a)], 'reshape_test_op', block_instance_name='reshape_test_op')

    # Test some basic methods related to blocks
    assert input_op.is_composite
    block_primitive = input_op.root_function.find_by_name('reshape_test_op')
    assert block_primitive.name == 'reshape_test_op'
    assert block_primitive.is_primitive
    assert block_primitive.is_block
    element_times_inside_block = block_primitive.block_root.find_by_name('element_times_inside_block')
    assert element_times_inside_block.name == 'element_times_inside_block'
    assert element_times_inside_block.is_primitive
    block_arguments_map = block_primitive.block_arguments_mapping
    assert len(block_arguments_map) == 1

    expected_forward = [[input_reshaped**2]]
    expected_backward = {a: input_tensor}

    # create batch
    input_tensor.shape = (1, 1) + input_tensor.shape

    forward_input = {a: input_tensor}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


def test_combine_op_as_block():
    # We test using combine as the operation that is encapsulated in a block
    from .. import combine, placeholder_variable, as_block, input_variable

    f = combine([placeholder_variable()])
    f = as_block(f, [(f.placeholders[0], placeholder_variable())], 'id')

    x = placeholder_variable()
    y = placeholder_variable()

    x = f.clone('share', {f.placeholders[0]: x})
    z = x - y

    # connect to inputs
    z.replace_placeholders({z.placeholders[0]: input_variable(1), z.placeholders[1]: input_variable(1)})

    # evaluate
    res = z.eval({z.arguments[0]: [[5.0]], z.arguments[1]: [[3.0]]})

    expected_forward = [[[2.]]]
    assert np.array_equal(res, expected_forward)


def test_block_with_duplicate_inputs():
    from .. import placeholder_variable, as_block, input_variable
    input = input_variable((1,), name='input')
    
    left_operand_placeholder = placeholder_variable(name='left_placeholder')
    right_operand_placeholder = placeholder_variable()
    plus_block = as_block(right_operand_placeholder + left_operand_placeholder, [(left_operand_placeholder, input), (right_operand_placeholder, input)], 'plus')

    plus_block_clone = plus_block.clone('share')


def test_as_block_with_function_in_arguments_map():
    from .. import placeholder_variable, as_block, input_variable
    input = input_variable((1,), name='input')
    input_plus_2 = input + 2
    
    left_operand_placeholder = placeholder_variable(name='left_placeholder')
    right_operand_placeholder = placeholder_variable()
    plus_block = as_block(right_operand_placeholder + left_operand_placeholder, [(left_operand_placeholder, input_plus_2), (right_operand_placeholder, input)], 'plus')

    # evaluate
    res = plus_block.eval({plus_block.arguments[0]: [[1.0]]})

    expected_forward = [[[4.]]]
    assert np.array_equal(res, expected_forward)
