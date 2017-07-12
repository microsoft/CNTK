# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


import numpy as np
import cntk as C

import pytest

def test_slice_stride():
    c = C.constant(value=list(range(0, 10)))
    assert np.all(c[0:3:2].eval() == [0, 2])

def test_eval_scalar():
    c = C.constant(value=2)
    assert (c+3).eval() == 5.0
    assert np.all((c+[3,4]).eval() == [5,6])

def test_numpy_conversion():
    from cntk.internal import sanitize_value
    from ..cntk_py import Value

    # check NDArrayView
    ndav = sanitize_value((2,3), 1, np.float32, None)
    assert np.all(ndav.asarray() == np.ones((2,3)))

    # check Value
    assert np.all(Value(ndav).asarray() == np.ones((2,3)))

    # check Constant
    c = C.constant(1, shape=(2,3))
    assert np.all(c.asarray() == np.ones((2,3)))
    
    #check Parameter
    p = C.parameter(shape=(2,3), init=1)
    assert np.all(p.asarray() == np.ones((2,3)))
