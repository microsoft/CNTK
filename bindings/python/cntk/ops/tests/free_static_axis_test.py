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
    x = C.input((C.FreeDimension, 2))
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
    x = C.sequence.input((C.FreeDimension, 2))
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
