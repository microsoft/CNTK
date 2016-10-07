# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy
import pytest

from cntk import DeviceDescriptor
from cntk.tests.test_utils import precision, PRECISION_TO_TYPE
from cntk.ops import *
from cntk.utils import *

# Keeping things short
AA = np.asarray
C = constant

# TOOD: adapt to v2 when needed


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


@pytest.mark.parametrize("data, expected", [
    ([], False),
    ([1], False),
    ([[1, 2]], False),
    ([[]], False),
    ([[AA([1, 2])]], False),
    ([AA([1, 2])], True),
    ([AA([1, 2]), AA([])], True),
])
def test_is_tensor_list(data, expected):
    assert is_tensor_list(data) == expected

def test_get_data_type():
    assert get_data_type(constant(value=2), constant(value=1)) == np.float32
    assert get_data_type(input_variable(shape=(2,3)), constant(value=1)) == np.float32

    ndav32 = create_NDArrayView_from_NumPy(np.asarray([[1,2]], dtype=np.float32))
    assert get_data_type(input_variable(shape=(2,3), data_type=np.float64),
            ndav32) == np.float64

    ndav64 = create_NDArrayView_from_NumPy(np.asarray([[1,2]],
        dtype=np.float64))
    assert get_data_type(input_variable(shape=(2,3), data_type=np.float64),
            ndav64) == np.float64

    val32 = create_Value_from_NumPy(np.asarray([[1,2]], dtype=np.float32),
            dev=DeviceDescriptor.default_device())
    assert get_data_type(val32, ndav64) == np.float64
