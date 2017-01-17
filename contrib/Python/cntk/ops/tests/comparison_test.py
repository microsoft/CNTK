# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for comparison operations, each operation is tested for 
the forward and the backward pass.
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision
from ...graph import *
from .. import *
from ...reader import *
import numpy as np

# we reuse same code for all comparsion functions 
def compasion_test_helper(left_operand, right_operand, func_to_test, func_computing_expected, device_id, precision):
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[ func_computing_expected(AA(left_operand), AA(right_operand))]]

    a = I([left_operand])
    b = I([right_operand])
    c = constant(left_operand)
    d = constant(right_operand)

    unittest_helper(func_to_test(a, b), None, expected, device_id=device_id,
                     precision=precision, clean_up=True, backward_pass=False)

    # Backward pass test
    #===================
    # the expected results for the backward pass is all zeroes
    left_as_input = func_to_test(a, d)

    right_as_input = func_to_test(c, b)

    expected = [[[np.zeros_like(x) for x in left_operand]]]
    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=a)
    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True, input_node=b)


TENSOR_PAIRS = [
    ([41., 42., 43., 42., 42., 42.], [42., 42., 42., 41., 42., 43.]),
    ]


# test for all comparison operations
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_less(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, less, np.less, device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_equal(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, equal, np.equal, device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_greater(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, greater, np.greater, device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_greater_equal(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, greater_equal, np.greater_equal, device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_not_equal(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, not_equal, np.not_equal, device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_less_equal(left_operand, right_operand, device_id, precision):
    compasion_test_helper(left_operand, right_operand, less_equal, np.less_equal, device_id, precision)
