# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for operations that are not differentiable.
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, precision, PRECISION_TO_TYPE

TENSORS = [
    ([12.3, -12.3]),
    ([10.2, -10.2]),
    ([0.5, -0.5]),
    ([0.01, -0.01]),
    ([0.499, -0.499]),
    ([5.0, -5.0]),
    ([0.0]),
    ([[2.1, 9.9], [4.7, 5.3]])
]


@pytest.mark.parametrize("operand", TENSORS)
def test_op_floor(operand, device_id, precision):
    operand = AA(operand)
    expected = np.floor(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import floor
    _test_unary_op(precision, device_id, floor, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_ceil(operand, device_id, precision):
    operand = AA(operand)
    expected = np.ceil(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import ceil
    _test_unary_op(precision, device_id, ceil, operand,
                   expected_forward, expected_backward)

# Manually setting the expectation since CNTK's round behaves differently than
# NumPy's round (see operator's docstring).
ROUND_TENSORS = [
    ([0.2, 1.3, 4.0, 5.5, 0.0],
     [0.0, 1.0, 4.0, 6.0, 0.0]),

    ([[0.6, 3.3], [1.9, 5.6]],
     [[1.0, 3.0], [2.0, 6.0]]),

    ([-5.5, -4.2, -3., -0.7, 0],
     [-5.0, -4.0, -3., -1.0, 0]),

    ([[-0.6, -4.3], [1.9, -3.2]],
     [[-1.0, -4.0], [2.0, -3.0]]),

    # CNTK is always rounding up values starting at x.5, while numpy rounds
    # to the nearest even value for half-integers
    # Refer here: https://en.wikipedia.org/wiki/Rounding#Tie-breaking
    # This test shows such values are not equal comparing numpy and CNTK
    ([0.5, 1.5, 2.5, 3.5],
     # NumPy would round to
     #        [0.0, 2.0, 2.0, 4.0]))
     # while CNTK rounds to
     [1.0, 2.0, 3.0, 4.0])
]


@pytest.mark.parametrize("operand,expected", ROUND_TENSORS)
def test_op_round(operand, expected, device_id, precision):
    operand, expected = AA(operand), AA(expected)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import round
    _test_unary_op(precision, device_id, round, operand,
                   expected_forward, expected_backward)

def test_input_variable():
    from .. import input, sequence
    i = input(shape=(2,3), name='i')
    assert i.shape == (2,3)
    assert i.name == 'i'
    assert len(i.dynamic_axes)==1

    sequence_i = sequence.input(shape=(3,2), name='sequence_i')
    assert sequence_i.shape == (3,2)
    assert sequence_i.name == 'sequence_i'
    assert len(sequence_i.dynamic_axes)==2

