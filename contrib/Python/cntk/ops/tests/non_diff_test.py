# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation is tested for
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision
from ...graph import *
from ...reader import *
import numpy as np
from .. import ceil, floor, round

# Testing inputs
@pytest.mark.parametrize("arg", [([12.3, -12.3]), ([10.2, -10.2]), ([0.5, -0.5]), ([0.01, -0.01]), ([0.499, -0.499]), ([5.0, -5.0]), ([0.0]), ([[2.1, 9.9], [4.7, 5.3]])])
def test_op_floor(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    numArray = [[AA(arg)]]
    exp = np.floor(numArray)

    a = I([arg], dynamic_axis='')
    op = floor(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # the expected results for the backward pass is all zeroes
    expected = [[[np.zeros_like(x) for x in arg]]]
    unittest_helper(op, None, expected, device_id, precision, clean_up=True, backward_pass=True, input_node=a)

@pytest.mark.parametrize("arg", [([12.3, -12.3]), ([10.2, -10.2]), ([0.5, -0.5]), ([0.01, -0.01]), ([0.499, -0.499]), ([5.0, -5.0]), ([0.0]), ([[2.1, 9.9], [4.7, 5.3]])])
def test_op_ceil(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    numArray = [[AA(arg)]]
    exp = np.ceil(numArray)

    a = I([arg], dynamic_axis='')
    op = ceil(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # the expected results for the backward pass is all zeroes
    expected = [[[np.zeros_like(x) for x in arg]]]
    unittest_helper(op, None, expected, device_id, precision, clean_up=True, backward_pass=True, input_node=a)

@pytest.mark.parametrize("arg", [([12.3, -12.3]), ([10.2, -10.2]), ([0.01, -0.01]), ([0.499, -0.499]), ([5.0, -5.0]), ([0.0]), ([[2.1, 9.9], [4.7, 5.3]])])
def test_op_round(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    # Refere to test test_op_roundnonstandard for values in form of x.5
    numArray = [[AA(arg)]]
    exp = np.round(numArray)

    a = I([arg], dynamic_axis='')
    op = round(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # the expected results for the backward pass is all zeroes
    expected = [[[np.zeros_like(x) for x in arg]]]
    unittest_helper(op, None, expected, device_id, precision, clean_up=True, backward_pass=True, input_node=a)

def test_op_roundnonstandard(device_id, precision):

    # Non-standard round values test
    # ==================
    # CNTK is doing round up for values like x.5, while numpy rounds to the nearest even value for half-integers
    # Refer here: https://en.wikipedia.org/wiki/Rounding#Tie-breaking
    # This test shows such values are not equal comparing numpy and CNTK
    arg = [([0.5, 1.5, 2.5, 3.5])]
    a = I([arg], dynamic_axis='')
    numpy_expected = [([0.0, 2.0, 2.0, 4.0])]
    op = round(a)
    cntk_expected = [np.array([[[1., 2., 3., 4.]]])]

    np.testing.assert_array_equal(np.round(arg),numpy_expected)
    unittest_helper(op, None, cntk_expected, device_id, precision, clean_up=True, backward_pass=False)
