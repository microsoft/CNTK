# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import pytest
import scipy.sparse as sparse
csr = sparse.csr_matrix

from ..core import *
from cntk.tests.test_utils import *
from cntk.ops.tests.ops_test_utils import compare_lists_of_np_arrays
from cntk import *
from cntk import asarray, asvalue

test_numbers = [4., 5, 6., 7., 8.]
test_array = AA(test_numbers, dtype=np.float32)

def _dense_value_to_ndarray_test(data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes):
    shape = (5,)

    if num_of_dynamic_axes == 2:
        var = input_variable(shape)
    elif num_of_dynamic_axes == 1:
        var = input_variable(shape, dynamic_axes=[Axis.default_batch_axis()])
    else:
        var = input_variable(shape, dynamic_axes=[])

    # conversion array -> value
    val = asvalue(var, data)
    assert val.shape == expected_value_shape

    # conversion value -> array
    dense_result = asarray(var, val)

    if isinstance(data, list):
        result_shapes = [AA(v).shape for v in dense_result]
    else:
        result_shapes = dense_result.shape

    assert result_shapes == expected_array_shapes

def _sparse_value_to_csr_test(data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes):
    shape = (3,)

    if num_of_dynamic_axes == 2:
        var = input_variable(shape, is_sparse=True)
    elif num_of_dynamic_axes == 1:
        var = input_variable(shape, is_sparse=True, dynamic_axes=[Axis.default_batch_axis()])
    else:
        var = input_variable(shape, is_sparse=True, dynamic_axes=[])

    # conversion csr array -> value
    val = asvalue(var, data)

    assert val.shape == expected_value_shape

    # conversion value -> csr array
    csr_result = asarray(var, val)

    csr_result_shapes = [v.shape for v in csr_result]

    assert csr_result_shapes == expected_csr_shapes

DENSE_CONFIGURATIONS = [
    # (dense data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)
    ([[test_array],
      [test_array, test_array]], 2, (2,2,5), [(1,5),(2,5)]),
    ([test_array,
      test_array], 2, (2, 1, 5), [(1,5), (1,5)]),
    ([[test_array],
      [test_array]], 2, (2, 1, 5), [(1,5), (1,5)]),
    (test_array, 2, (5,), (5,)),
    (AA([test_numbers], dtype=np.float32), 2, (1,5), (1,5)),
    (AA([test_numbers, test_numbers], dtype=np.float32), 2, (2,5), (2,5)),
    ([test_array,
      test_array], 1, (2,1,5), [(1,5), (1,5)]),
    ([[test_array],
      [test_array]], 1, (2,1,5), [(1,5), (1,5)]),
    (AA([test_numbers, test_numbers], dtype=np.float32), 1, (2,5), (2,5)),
    (AA([test_numbers], dtype=np.float32), 1, (1,5), (1,5)),
    ([test_array,
      test_array], 0, (2,5), [(5,), (5,)]),
    (AA([test_numbers, test_numbers], dtype=np.float32), 0, (2,5), (2,5)),
    (test_array, 0, (5,), (5,)),
]

@pytest.mark.parametrize("data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes", DENSE_CONFIGURATIONS)
def test_dense_value_to_ndarray(data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes):
    _dense_value_to_ndarray_test(data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)

SPARSE_ARRAYS = [
    # (sparse data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)
    ([csr([[1.,0.,2.], [2.,3.,0.]]),
      csr([5.,0.,1.])], 2, (2, 2, 3), [(2,3),(1,3)]),
    ([csr([1,0,2]),
      csr([5,0,1])], 2, (2, 1, 3),[(1,3),(1,3)]),
    ([csr([[1,0,2],[2,3,4]])], 2, (1, 2, 3), [(2,3)]),
    ([csr([1,0,2]),
      csr([5,0,1])], 1, (2, 1, 3), [(1,3),(1,3)]),
    ([csr([[1,0,2], [2,3,0]]),
      csr([[5,0,1], [2,3,0]])], 1, (2, 2, 3), [(2,3),(2,3)]),
    ([csr([[1,0,2],[2,3,4]])], 1, (1, 2, 3), [(2,3)]),
    (csr([1,0,2]), 0, (1, 3), [(1,3)]),
]

@pytest.mark.parametrize("data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes", SPARSE_ARRAYS)
def test_sparse_value_to_csr(data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes):
    _sparse_value_to_csr_test(data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes)

DENSE_FAILING_CONFIGURATIONS = [
    # (dense data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)
    ([[test_array],
     [test_array, test_array]], 0, (2,2,5), [(1,5),(2,5)]),
    # TODO: enable once check is implemented
    #([[test_array],
    #  [test_array]], 0, (2, 1, 5), [(1, 5),(1, 5)]),
    #([[test_array],
    #  [test_array, test_array]], 1, (2,2,5), [(1,5),(2,5)]),
]

SPARSE_FAILING_CONFIGURATIONS = [
    # (sparse data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)
    (csr([[1,0,2], [2,3,0]]), 2, (1, 3), [(1,3)]),
    (csr([[1,0,2],[2,3,4]]), 2, (2, 1, 3), [(1,3),(1,3)]),
    (csr([[1,0,2], [2,3,0]]), 1, (1, 3), [(1,3)]),
    ([csr([[1,0,2],[2,3,4]])], 0, (1, 2, 3), [(2,3)]),
    ([csr([[1,0,2], [2,3,0]]),
      csr([5,0,1])], 0, (2, 2, 3), [(2,3),(1,3)]),
    ([csr([1,0,2])], 0, (1, 3), [(1,3)]),
    # TODO: enable once check is implemented
    #([csr([[1,0,2], [2,3,0]]),
    #  csr([5,0,1])], 1, (2, 2, 3), [(2,3),(1,3)]),
]

@pytest.mark.parametrize("data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes", DENSE_FAILING_CONFIGURATIONS)
def test_dense_failing_value_to_ndarray(data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes):
    with pytest.raises(ValueError):
        _dense_value_to_ndarray_test(data, num_of_dynamic_axes, expected_value_shape, expected_array_shapes)

@pytest.mark.parametrize("data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes", SPARSE_FAILING_CONFIGURATIONS)
def test_sparse_failing_value_to_csr(data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes):
    with pytest.raises(ValueError):
        _sparse_value_to_csr_test(data, num_of_dynamic_axes, expected_value_shape, expected_csr_shapes)
