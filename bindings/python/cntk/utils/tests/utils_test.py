# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import scipy.sparse as sparse
csr = sparse.csr_matrix
import pytest

from cntk.tests.test_utils import precision, PRECISION_TO_TYPE
from cntk.ops import *
from cntk.utils import *
from cntk.internal import *
from cntk import Value

AA = np.asarray

def test_sanitize_dtype_numpy():
    for dtype in ['float', 'float32', np.float32, int]:
        assert sanitize_dtype_numpy(dtype) == np.float32, dtype
    for dtype in [float, 'float64', np.float64]:
        assert sanitize_dtype_numpy(dtype) == np.float64, dtype

def test_sanitize_dtype_cntk():
    for dtype in ['float', 'float32', np.float32, int]:
        assert sanitize_dtype_cntk(dtype) == cntk_py.DataType_Float, dtype
    for dtype in [float, 'float64', np.float64]:
        assert sanitize_dtype_cntk(dtype) == cntk_py.DataType_Double, dtype

@pytest.mark.parametrize("data, dtype", [
    ([1], np.float32),
    ([[1, 2]], np.float64),
    (2, np.float64),
    (AA([1,2], dtype=np.float32), np.float64),
])
def test_sanitize_input(data, dtype):
    inp = sanitize_input(data, dtype)
    assert np.allclose(inp.value, data)
    assert inp.dtype == dtype

def test_axes():
    axes = [Axis.default_batch_axis(), Axis.default_dynamic_axis()]
    assert tuple(axes) == Axis.default_input_variable_dynamic_axes()
    assert sanitize_dynamic_axes(axes) == \
            tuple(reversed(Axis.default_input_variable_dynamic_axes()))

    assert (Axis.default_dynamic_axis(),) == \
            sanitize_dynamic_axes(Axis.default_dynamic_axis())

def test_get_data_type():
    pa32 = parameter(init=np.asarray(2, dtype=np.float32))
    pa64 = parameter(init=np.asarray(2, dtype=np.float64))
    pl = placeholder(shape=(2))
    c = constant(value=3.0)
    n32 = AA(1, dtype=np.float32)
    n64 = AA(1, dtype=np.float64)

    assert get_data_type(pa32) == np.float32
    assert get_data_type(pa32, n32) == np.float32
    assert get_data_type(n32, n32) == np.float32
    assert get_data_type(n32, n64) == np.float64
    assert get_data_type(pl, n64) == np.float64
    assert get_data_type(pl, n32) == np.float32
    assert get_data_type(pl, pl) == None
    # variable's type shall take precedence over provided data
    assert get_data_type(pa32, n64) == np.float32
    assert get_data_type(pa64, n64) == np.float64
    assert get_data_type(pa32, pl, n64) == np.float32
    assert get_data_type(pa64, pl, n64) == np.float64

def test_sanitize_batch_sparse():
    batch = [csr([[1,0,2],[2,3,0]]),
             csr([5,0,1])]

    var = sequence.input(3, is_sparse=True)
    b = sanitize_batch(var, batch)
    # 2 sequences, with max seq len of 2 and dimension 3
    assert b.shape == (2,2,3)

@pytest.mark.parametrize("batch, seq_starts, expected", [
    ([AA([5, 6, 7]), AA([8])],
       [True, False],
       [[2, 1, 1], [1, 0, 0]]),

    ([AA([5]), AA([8])],
       [True, False],
       [[2], [1]]),

    ([[5, 6, 7], [8]],
       [True, False],
       [[2, 1, 1], [1, 0, 0]]),

    (Value.one_hot([[3, 4, 5, 1], [60, 61]], num_classes=62),
        [True, False],
        ValueError),
])
def test_mask(batch, seq_starts, expected):
    shape = ()
    var = sequence.input(shape)
    if type(expected) == type(ValueError):
        with pytest.raises(expected):
            s = sanitize_batch(var, batch, seq_starts)
    else:
        s = sanitize_batch(var, batch, seq_starts)
        assert np.allclose(s.mask, expected)

def test_one_hot():
    with pytest.raises(ValueError):
        s = Value.one_hot([[1.0, 2.0], [3.]], 4)
    with pytest.raises(ValueError):
        s = Value.one_hot([1, 2], 4)

def test_sanitize_batch_contiguity():
    a1 = AA([[1,2],[3,4]])
    a2 = AA([[5,6],[7,8]])
    var = sequence.input((2,2), is_sparse=True)

    batch = [a1.T,a2.T]
    with pytest.warns(RuntimeWarning):
        b = sanitize_batch(var, batch)
        assert b.shape == (2,1,2,2)

    batch = [a1,a2]
    b = sanitize_batch(var, batch)
    assert b.shape == (2,1,2,2)

