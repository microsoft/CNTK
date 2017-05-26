# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for comparison operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE

TENSOR_PAIRS = [
    ([41., 42., 43., 42., 42., 42.], [42., 42., 42., 41., 42., 43.]),
]

from cntk import equal, less, less_equal, greater, greater_equal, not_equal

FUNCTIONS_TO_TEST = [
    (equal, np.equal),
    (less, np.less),
    (less_equal, np.less_equal),
    (greater, np.greater),
    (greater_equal, np.greater_equal),
    (not_equal, np.not_equal),
]

test_parameters = []
import itertools as itt
for functions_to_test, tensor_pairs in itt.product(FUNCTIONS_TO_TEST, TENSOR_PAIRS):
    cntk_func, numpy_func = functions_to_test
    left_op, right_op = tensor_pairs
    test_parameters.append((cntk_func, numpy_func, left_op, right_op))


@pytest.mark.parametrize("cntk_function, numpy_function, left_operand, right_operand", test_parameters)
def test_op_comparison(left_operand, right_operand, cntk_function, numpy_function, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    expected_forward = [numpy_function(
        AA(left_operand, dtype=dt), AA(right_operand, dtype=dt))]

    expected_backward = {
        'left_arg':  [np.zeros_like(left_operand, dtype=dt)],
        'right_arg': [np.zeros_like(left_operand, dtype=dt)]
    }

    _test_binary_op(precision, device_id, cntk_function, left_operand, right_operand,
                    expected_forward, expected_backward)
