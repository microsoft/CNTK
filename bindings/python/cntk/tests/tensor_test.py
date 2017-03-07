# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..tensor import *
from ..ops import constant, parameter
import numpy as np

import pytest

def test_overload_exception():
    c = constant(value=list(range(0, 10)))

    with pytest.raises(ValueError):
        c[0:3:2]

def test_eval_scalar():
    c = constant(value=2)
    assert (c+3).eval() == 5.0
    assert np.all((c+[3,4]).eval() == [5,6])

def test_numpy_conversion():
    from cntk.internal import sanitize_value
    from ..cntk_py import Value

    # check NDArrayView
    ndav = sanitize_value((2,3), 1, np.float32, None)
    assert np.all(np.asarray(ndav) == np.ones((2,3)))

    # check Value
    assert np.all(np.asarray(Value(ndav)) == np.ones((2,3)))

    # check Constant
    c = constant(1, shape=(2,3))
    assert np.all(np.asarray(c) == np.ones((2,3)))
    
    #check Parameter
    p = parameter(shape=(2,3), init=1)
    assert np.all(np.asarray(p) == np.ones((2,3)))
    
