# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for combine operation, only forward pass is tested
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import AA, I, precision, PRECISION_TO_TYPE, compare_lists_of_np_arrays, cntk_device
from ...utils import sanitize_dtype_cntk, eval as cntk_eval

from .. import plus, minus, classification_error, cross_entropy_with_softmax

TENSOR_PAIRS = [
    # (first operand, second_operand, ops, expected_forward)
    ([[1., 2., 3., 4., 5]], [[0., 0., 0., 1., 0.]], [plus, minus],
     [[[1., 2., 3., 5., 5]],[[1., 2., 3., 3., 5]]]),
    ([[1., 2., 3., 4., 5]], [[0., 0., 0., 1., 0.]], [cross_entropy_with_softmax, classification_error],
     [[[1.]],[[1.451914]]])
]

@pytest.mark.parametrize("left_operand, right_operand, operations, expected_results", TENSOR_PAIRS)
def test_op_combine(left_operand, right_operand, operations, expected_results, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    from .. import combine

    left_value = AA(left_operand, dtype=dt)
    right_value = AA(right_operand, dtype=dt)

    a = I(shape=left_value.shape,
          dtype=sanitize_dtype_cntk(precision),
          needs_gradient=True,
          name='a')

    b = I(shape=right_value.shape,
          dtype=sanitize_dtype_cntk(precision),
          needs_gradient=True,
          name='b')

    left_value.shape = (1, 1) + left_value.shape
    right_value.shape = (1, 1) + right_value.shape

    forward_input = {a: left_value, b: right_value}

    combine_list = []
    for op in operations:
        combine_list.append(op(a,b))

    combine_node = combine(combine_list)

    expected_forward_results = [np.asarray([[i]], dtype=dt) for i in expected_results]

    forward_results, _ = cntk_eval(combine_node, forward_input, precision,
            cntk_device(device_id))

    results = list(forward_results.values())

    assert compare_lists_of_np_arrays(results, expected_forward_results)


def test_op_combine_input_var():
    from .. import combine, input_variable

    x = input_variable(shape=(2))
    func = combine([x])
    value = [[1, 2]]
    res = func.eval({x : value})
    
    assert np.allclose(res, [[1, 2]])
