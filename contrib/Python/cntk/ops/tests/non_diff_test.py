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
from .ops_test_utils import unittest_helper, C, AA, I, precision
from ...graph import *
from ...reader import *
import numpy as np
from ..non_diff import floor

# Testing inputs
@pytest.mark.parametrize("arg", [([12.3,-12.3]),([10.2,-10.2]),([0.5,-0.5]),([0.01,-0.01]),([0.499,-0.499]),([5.0,-5.0]),([0.0])])
def test_op_floor(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    numArray = [[AA(arg)]]
    exp = np.floor(numArray)

    a = I([arg], has_sequence_dimension=False)
    op = Floor(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=False, backward_pass=False)

    # Backward pass test
    # ==================
    # the expected results for the backward pass is all zeroes
    expected = [[[np.zeros_like(x) for x in arg]]]
    unittest_helper(op, None, expected, device_id, precision, clean_up=True, backward_pass=True, input_node=a)

@pytest.mark.parametrize("arg", [([12.3,-12.3]),([10.2,-10.2]),([0.5,-0.5]),([0.01,-0.01]),([0.499,-0.499]),([5.0,-5.0]),([0.0])])
def test_op_ceil(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    numArray = [[AA(arg)]]
    exp = np.ceil(numArray)

    a = I([arg], has_sequence_dimension=False)
    op = Ceil(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=False, backward_pass=False)

    # Backward pass test
    # ==================
    # the expected results for the backward pass is all zeroes
    expected = [[[np.zeros_like(x) for x in arg]]]
    unittest_helper(op, None, expected, device_id, precision, clean_up=True, backward_pass=True, input_node=a)

@pytest.mark.parametrize("arg", [([12.3,-12.3]),([10.2,-10.2]),([0.01,-0.01]),([0.499,-0.499]),([5.0,-5.0]),([0.0])])
def test_op_round(arg, device_id, precision):

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample
    # Refere to test test_op_roundnonstandard for values in form of x.5
    numArray = [[AA(arg)]]
    exp = np.round(numArray)

    a = I([arg], has_sequence_dimension=False)
    op = Round(a)
    unittest_helper(op, None, exp, device_id, precision, clean_up=False, backward_pass=False)

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
    arg = 0.5

    numArray = [[AA(arg)]]
    cntk_exp = Round(arg)

    assert not np.round(numArray) == cntk_exp