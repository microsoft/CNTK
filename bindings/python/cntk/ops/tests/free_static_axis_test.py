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

