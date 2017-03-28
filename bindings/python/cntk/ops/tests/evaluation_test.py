# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evaluation operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

from cntk import input, dropout, combine


def test_sequence_grad_as_numpy_false(device_id, precision):
    from .. import sequence

    a = sequence.input(shape=(1,), dtype=PRECISION_TO_TYPE[precision], needs_gradient=True, name='a')

    sequence_sum_a_plus_sequence_sum_a = sequence.reduce_sum(a) + sequence.reduce_sum(a)

    a_data = [AA([[2]], dtype=PRECISION_TO_TYPE[precision]), AA([[2], [3]], dtype=PRECISION_TO_TYPE[precision]), AA([[2], [3], [4]], dtype=PRECISION_TO_TYPE[precision])]

    actual_grad = sequence_sum_a_plus_sequence_sum_a.grad({a: a_data}, [a], as_numpy=False)
    
    test_op = a + 1
    result = test_op.eval({a : actual_grad})
    assert np.array_equal(result[0], np.asarray([[3.]]))
    assert np.array_equal(result[1], np.asarray([[3.], [3.]]))
    assert np.array_equal(result[2], np.asarray([[3.], [3.], [3.]]))

def test_grad_with_no_arguments_needing_gradients():
    x = input(10)
    z = dropout(x, .4)
    with pytest.raises(ValueError):
        _, result = z.grad({x: [np.array([5]*150, "float32").reshape(15, 10)]}, outputs=[z])

def test_eval_not_all_outputs():
    x = input(1)
    x_data = [AA([3], dtype=np.float32)]
    y = input(1)
    y_data = [AA([2], dtype=np.float32)]
    plus_func = x + 1
    minus_func = y - 1
    func = combine([plus_func, minus_func])

    result = func.eval({x : x_data}, [plus_func])
    assert np.array_equal(result, np.asarray([[4.]]))

    result = func.eval({y : y_data}, [minus_func])
    assert np.array_equal(result, np.asarray([[1.]]))
