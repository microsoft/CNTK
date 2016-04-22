
import numpy
import pytest

from cntk.tests.test_utils import unittest_helper, precision, PRECISION_TO_TYPE
from cntk.ops.variables_and_parameters import *
from cntk.utils import *

#Keeping things short
AA = np.asarray
C = constant
I = input_array

@pytest.mark.parametrize("alias, idx, data, expected", [
    ('', 0, [AA([1, 0]), AA([0, 0, 1, 0])], ValueError),  # no alias given
    ('A', 0, [object()], ValueError),
])
def test_tensor_conversion_exceptions(alias, idx, data, expected):
    with pytest.raises(expected):
        tensor_to_text_format(idx, alias, data)


@pytest.mark.parametrize("alias, idx, data, expected", [
    ('W', 0, AA([]), "0\t|W "),
    ('W', 0, AA([[1, 0, 0, 0], [1, 0, 0, 0]]), """\
0\t|W 1 1 0 0 0 0 0 0\
"""),
])
def test_tensor_conversion_dense(alias, idx, data, expected):
    assert tensor_to_text_format(idx, alias, data,
            has_sequence_dimension=False) == expected

@pytest.mark.parametrize("data, expected", [
    ([], True),
    ([1], True),
    ([[1, 2]], True),
    ([[]], True),
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

