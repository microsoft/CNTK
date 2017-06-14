# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the ability to leave some static axes of an input free
and inferred from the bound value during compute.
"""

import numpy as np
import pytest
import cntk as C

def test_free_static_dimension_basic():
    x = C.input_variable((C.FreeDimension, 2))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))


def test_free_static_axis_in_recurrence():
    x = C.sequence.input_variable((C.FreeDimension, 2))
    out_placeholder = C.placeholder()
    out_past = C.sequence.past_value(out_placeholder)
    wh = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    wx = C.parameter(init=np.asarray([[1, 4], [2, 5]], dtype=np.float32))
    out = C.times(x, wx) + C.times(out_past, wh)
    out.replace_placeholders({out_placeholder : out})
    
    x_data = np.asarray([[0.5, 0.2], [-0.7, 1.2]], np.float32)
    w_grad, out_val = out.grad({x : x_data}, wrt=[wh, wx], outputs=[out])
    assert np.allclose(out_val, [[[[0.9, 3.], [1.7, 3.2]]]])
    assert np.allclose(w_grad[wx], [[-0.2, -0.2], [1.4, 1.4]])


def test_reshape_free_static_axis():
    x = C.input_variable((C.FreeDimension, 2, 3))
    x_reshaped = C.reshape(x, (-1), 0, 2)
    assert x_reshaped.shape == (C.FreeDimension, 3)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(4, 3))

    x_data = np.arange(18).reshape(3, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(6, 3))

    x_reshaped = C.reshape(x, (-1), 1, 3)
    assert x_reshaped.shape == (C.FreeDimension, 6)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(2, 6))

    x_reshaped = C.reshape(x, (4), 0, 2)
    assert x_reshaped.shape == (4, 3)
    x_data = np.arange(12).reshape(2, 2, 3)
    result = x_reshaped.eval({x : x_data})
    assert np.array_equal(result[0], x_data.reshape(4, 3))

    x_data = np.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        result = x_reshaped.eval({x : x_data})
   

def test_free_and_inferred_static_dimension():
    x = C.input_variable((C.FreeDimension, -1))
    w = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    t = C.times(x, w)

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2], [0.1, .6]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.allclose(t_val, np.asarray([[[1.2, 3.1], [0.8, 2.3]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.6, .6], [.8, .8]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2]], np.float32)
    w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])
    assert np.array_equal(t_val, np.asarray([[[1.2, 3.1]]], dtype=np.float32))
    assert np.array_equal(w_grad, np.asarray([[0.5, .5], [.2, .2]], dtype=np.float32))

    x_data = np.asarray([[0.5, 0.2, 0.9]], np.float32)
    with pytest.raises(ValueError):
        w_grad, t_val = t.grad({x : x_data}, wrt=[w], outputs=[t])


def test_inferred_static_axis_in_recurrence():
    x = C.sequence.input_variable((-1, 2))
    out_placeholder = C.placeholder()
    out_past = C.sequence.past_value(out_placeholder)
    wh = C.parameter(init=np.asarray([[2, 5], [1, 3]], dtype=np.float32))
    wx = C.parameter(init=np.asarray([[1, 4], [2, 5]], dtype=np.float32))
    out = C.times(x, wx) + C.times(out_past, wh)
    out.replace_placeholders({out_placeholder : out})
    
    x_data = np.asarray([[0.5, 0.2], [-0.7, 1.2]], np.float32)
    w_grad, out_val = out.grad({x : x_data}, wrt=[wh, wx], outputs=[out])
    assert np.allclose(out_val, [[[[0.9, 3.], [1.7, 3.2]]]])
    assert np.allclose(w_grad[wx], [[-0.2, -0.2], [1.4, 1.4]])


def test_slice_with_inferred_static_axis():
    x = C.input_variable(shape=(C.InferredDimension, C.InferredDimension, 3))
    padding_shape = (3, C.InferredDimension, 3)
    y = C.splice(C.constant(value=0, shape=padding_shape), x, axis=0)
    assert y.shape == (-1, -1, 3)
    y = C.splice(x, C.constant(value=0, shape=padding_shape), axis=0)
    assert y.shape == (-1, -1, 3)


def test_free_dimension_broadcast():
    i0 = C.sequence.input_variable(shape=(5,))
    i0_unpacked, _ = C.sequence.unpack(i0, padding_value=0).outputs
    i1 = C.input_variable(shape=(5,))
    m = i0_unpacked * i1
    assert m.shape == (-3, 5)

    i1 = C.input_variable(shape=(1,5,))
    m = i0_unpacked * i1
    assert m.shape == (-3, 5)
