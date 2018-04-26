# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the stop_gradient.
"""

import numpy as np
import cntk as C
from .ops_test_utils import AA, PRECISION_TO_TYPE, precision
from cntk.internal import sanitize_dtype_cntk
import pytest


def test_stop_gradient():
    x = C.sequence.input_variable(shape=(2,), sequence_axis=C.Axis("B"), needs_gradient=True)
    y = C.sequence.input_variable(shape=(2,), sequence_axis=C.Axis("B"), needs_gradient=True)
    z = C.element_times(x, y)
    w = z + C.stop_gradient(z)
    a = np.reshape(np.float32([0.25, 0.5, 0.1, 1]), (1, 2, 2))
    b = np.reshape(np.float32([-1.25, 1.5, 0.1, -1]), (1, 2, 2))
    bwd, fwd = w.forward({x: a, y: b}, [w.output], set([w.output]))
    value = list(fwd.values())[0]
    expected = np.multiply(a, b)*2
    assert np.allclose(value, expected)
    grad = w.backward(bwd, {w.output: np.ones_like(value)}, set([x, y]))
    assert np.allclose(grad[x], b)
    assert np.allclose(grad[y], a)

    #test stop_gradient with function as input whose arguments should have no gradients (zeros reading)
    w = C.stop_gradient(z)
    bwd, fwd = w.forward({x: a, y: b}, [w.output], set([w.output]))
    value = list(fwd.values())[0]
    expected = np.multiply(a, b)
    assert np.allclose(value, expected)
    grad = w.backward(bwd, {w.output: np.ones_like(value)}, set([x, y]))
    #there should be no gradients backward to x and y
    assert np.allclose(grad[x], np.zeros_like(b))
    assert np.allclose(grad[y], np.zeros_like(a))


NO_BACKPROP_TEST_OPERANDS = [
    # (input_data, )
    ([[1]], ),
    ([[1, 2], [4, 5]], ),
    ([[1, 2], [4, 5]], ),
    ([[1, 2], [4, 5]], ),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ),
]


@pytest.mark.parametrize("input_data", NO_BACKPROP_TEST_OPERANDS)
def test_stop_gradient_no_backprop(input_data, precision):
    no_backprop_ops = [C.stop_gradient]
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    x = C.input_variable(shape=data.shape,
              dtype=sanitize_dtype_cntk(dt),
              needs_gradient=True,
              name='a')
    w = C.parameter(x.shape, init=np.ones(x.shape).astype(dt) * 3.0)
    # create batch
    data.shape = (1,) + data.shape
    expected_x_backward = np.zeros_like(data)
    expected_w_backward = np.zeros_like(w)

    # numpy argmax doesn't support keepdims
    for op in no_backprop_ops:
        #test direct input: no gradients pass through to inputs
        op_func = op(x)
        grad = op_func.grad({x: data}, [x])
        np.testing.assert_almost_equal(grad, expected_x_backward)

        #test inputs through sub-expressions: no gradients pass through to inputs (e.g. x, w) of the subexpressoin (e.g. x * w here)
        op_func = op(x * w)
        grad = op_func.grad({x: data}, [w, x])
        np.testing.assert_almost_equal(grad[x], expected_x_backward)
        np.testing.assert_almost_equal(grad[w], expected_w_backward)

        #testing inputs through shared sub-expressions: no gradients pass through reduce arg ops to inputs (e.g. x, w) of the subexpressoin
        # (e.g. x * w here), therefore the gradients will depend on how the shared expressions participate in other experssions:
        shared_exp = x * w
        op_func = op(shared_exp) + x + w + shared_exp
        ref_op_func = x + w + shared_exp
        grad = op_func.grad({x: data}, [w, x])
        ref_grad = ref_op_func.grad({x: data}, [w, x])
        np.testing.assert_almost_equal(grad[x], ref_grad[x])
        np.testing.assert_almost_equal(grad[w], ref_grad[w])
