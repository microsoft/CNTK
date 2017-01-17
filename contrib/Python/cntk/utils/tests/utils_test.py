# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy
import pytest

from cntk.tests.test_utils import unittest_helper, precision, PRECISION_TO_TYPE
from cntk.ops import *
from cntk.utils import *

# Keeping things short
AA = np.asarray
C = constant
I = input_numpy


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
