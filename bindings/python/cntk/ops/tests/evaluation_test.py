# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evaluation operations (grad and eval)
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

from cntk import dropout, combine
import cntk as C

def test_sequence_grad_as_numpy_false(device_id, precision):
    from .. import sequence

    a = sequence.input_variable(shape=(1,), dtype=PRECISION_TO_TYPE[precision], needs_gradient=True, name='a')

    sequence_sum_a_plus_sequence_sum_a = sequence.reduce_sum(a) + sequence.reduce_sum(a)

    a_data = [AA([[2]], dtype=PRECISION_TO_TYPE[precision]), AA([[2], [3]], dtype=PRECISION_TO_TYPE[precision]), AA([[2], [3], [4]], dtype=PRECISION_TO_TYPE[precision])]

    actual_grad = sequence_sum_a_plus_sequence_sum_a.grad({a: a_data}, [a], as_numpy=False)
    
    test_op = a + 1
    result = test_op.eval({a : actual_grad})
    assert np.array_equal(result[0], np.asarray([[3.]]))
    assert np.array_equal(result[1], np.asarray([[3.], [3.]]))
    assert np.array_equal(result[2], np.asarray([[3.], [3.], [3.]]))

def test_grad_with_no_arguments_needing_gradients():
    x = C.input_variable(10)
    z = dropout(x, .4)
    with pytest.raises(ValueError):
        _, result = z.grad({x: [np.array([5]*150, "float32").reshape(15, 10)]}, outputs=[z])

def test_eval_not_all_outputs():
    x = C.input_variable(1)
    x_data = [AA([3], dtype=np.float32)]
    y = C.input_variable(1)
    y_data = [AA([2], dtype=np.float32)]
    plus_func = x + 1
    minus_func = y - 1
    func = combine([plus_func, minus_func])

    result = func.eval({x : x_data}, [plus_func])
    assert np.array_equal(result, np.asarray([[4.]]))

    result = func.eval({y : y_data}, [minus_func])
    assert np.array_equal(result, np.asarray([[1.]]))


def test_grad_custimized_root():
    x = C.input_variable(shape=(1,), needs_gradient=True)
    y = C.sqrt(x)
    y2 = C.log(x)
    combine = C.combine([y.output, y2.output])
    a = np.asarray([1,4,16], dtype=np.float32).reshape(3,1)
    grads = combine.grad({x:a}, grad_root = y.output)
    expect_grad = np.asarray([[0.5],[0.25],[0.125]], dtype=np.float32)
    assert np.array_equal(grads, expect_grad)


def test_constant_eval():
    c = C.Constant(value=1)
    c_plus_1 = c + 1
    op = C.combine([c_plus_1, c])
    result = op.eval({})
    assert np.array_equal(result[c_plus_1.output], 2.0)
    assert np.array_equal(result[c], 1.0)


def test_input_without_dynamic_axes():
    x = C.input_variable(shape=(2,), dynamic_axes=[], needs_gradient=True, name='x')
    assert len(x.dynamic_axes) == 0

    op = x * .01 + 3.0
    grad_result, eval_result = op.grad({x : np.asarray([.6, -.8], dtype=np.float32)}, outputs=[op], wrt=[x])
    assert np.allclose(eval_result, [3.006, 2.992])
    assert np.allclose(grad_result, [.01, .01])

    w = C.parameter(init=np.asarray([[0.5], [-1.5]], dtype=np.float32))
    op = C.times(x, w) + 3.0
    grad_result, eval_result = op.grad({x : np.asarray([.6, -.8], dtype=np.float32)}, outputs=[op], wrt=[w])
    assert np.allclose(eval_result, [4.5])
    assert np.allclose(grad_result, [[.6], [-.8]])


def test_grad_after_eval():
    x = C.input_variable((C.FreeDimension, 2))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    t_val = t.eval({x : x_data})
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))

    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    t_val = t.eval({x : x_data})
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))

    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))
    

def test_validation_before_eval():
    w = C.parameter((4,C.InferredDimension))
    v = C.parameter((C.InferredDimension,5))
    wv = C.times(w,v)

    p = C.input((4,1))
    wp = C.times(w,p)

    q = C.input((1,5))
    qv = C.times(q,v)

    with pytest.raises(ValueError):
        wv.eval()
