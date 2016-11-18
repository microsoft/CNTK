# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy
import scipy.sparse as sparse
csr = sparse.csr_matrix
import pytest

from cntk.device import default
from cntk.tests.test_utils import precision, PRECISION_TO_TYPE
from cntk.ops import *
from cntk.utils import *

# Keeping things short
AA = np.asarray
C = constant

@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'A': [object()]}, ValueError),
])
def test_tensor_conversion_exceptions(idx, alias_tensor_map, expected):
    with pytest.raises(expected):
        tensors_to_text_format(idx, alias_tensor_map)


@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'W': AA([])}, ""),
    (0, {'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]])}, """\
0\t|W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0], [1, 0]], [[5, 6], [7, 8]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 1 0
0\t|W 5 6 7 8"""),
])
def test_tensor_conversion_dense(idx, alias_tensor_map, expected):
    assert tensors_to_text_format(idx, alias_tensor_map) == expected


@pytest.mark.parametrize("data, expected", [
    ([1], True),
    ([[1, 2]], True),
    ([[AA([1, 2])]], False),
    ([AA([1, 2])], False),
    ([AA([1, 2]), AA([])], False),
])
def test_is_tensor(data, expected):
    assert is_tensor(data) == expected


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

def test_get_data_type():
    pa32 = parameter(init=np.asarray(2, dtype=np.float32))
    pa64 = parameter(init=np.asarray(2, dtype=np.float64))
    pl = placeholder_variable(shape=(2))
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

@pytest.mark.parametrize("shape, batch, expected", [
    (1, [[1,2]], True),
    (1, [1,2], False),

    (2, AA([[1,1],[2,2]]), False),
    (2, [[1,1],[2,2]], False),
    (2, AA([[[1,1],[2,2]]]), True),
    ((2,), AA([[1,1],[2,2]]), False),
    ((2,), AA([[[1,1],[2,2]]]), True),

    ((1,2), AA([[[1,1]],[[2,2]]]), False),
    ((1,2), AA([[[[1,1]],[[2,2]]]]), True),
    ((2,2), AA([[[1,1],[2,2]]]), False),
    ((2,2), AA([[[[1,1],[2,2]]]]), True),

    # exception handling
    ((2,2), AA([[1,1],[2,2]]), ValueError),
    (1, [[[1,2]]], ValueError),
    #(1, [AA([[40], [50]])], ValueError),
    ((1,), [[[40], [50]]], ValueError),
])
def test_has_seq_dim_dense(shape, batch, expected):
    i1 = input_variable(shape)
    if expected in [False, True]:
        assert has_seq_dim(i1, batch) == expected
    else:
        with pytest.raises(expected):
            has_seq_dim(i1, batch)

@pytest.mark.parametrize("shape, batch, expected", [
    ((1,2), [csr([1,0]), csr([2,3]), csr([5,6])], False),
    ((1,2), [[csr([1,0]), csr([2,3])], [csr([5,6])]], True),
])
def test_has_seq_dim_sparse(shape, batch, expected):
    i1 = input_variable(shape, is_sparse=True)
    if expected in [False, True]:
        assert has_seq_dim(i1, batch) == expected
    else:
        with pytest.raises(expected):
            has_seq_dim(i1, batch)

def test_sanitize_batch_sparse():
    batch = [[csr([1,0,2]), csr([2,3,0])],
             [csr([5,0,1])]]

    var = input_variable(3, is_sparse=True)
    b = sanitize_batch(var, batch)
    # 2 sequences, with max seq len of 2 and dimension 3
    assert b.shape == (2,2,3)

    var = input_variable((1,3), is_sparse=True)
    b = sanitize_batch(var, batch)
    # 2 sequences, with max seq len of 2 and dimension 3
    assert b.shape == (2,2,3)


