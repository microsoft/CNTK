# Here should all the functional operator tests go.

import numpy as np
import pytest
from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..reader import *

# keeping things short
C = constant
I = input
AA = np.asarray

def _test(root_node, expected, clean_up=True):
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        assert not ctx.input_nodes
        result = ctx.eval(root_node)
        expected = AA(expected)
        assert result.shape == expected.shape or result.shape == (
            1, 1) and expected.shape == ()
        assert np.allclose(result, expected)

_VALUES = [0, [[1, 2], [3, 4]], [10.1, -20.2], 1.1]

@pytest.fixture(scope="module", params=_VALUES)
def left_arg(request):
    return request.param

right_arg = left_arg

def test_op_add(left_arg, right_arg):
    expected = AA(left_arg) + AA(right_arg)
    _test(C(left_arg) + right_arg, expected)
    _test(C(left_arg) + C(right_arg), expected)
    _test(left_arg + C(right_arg), expected)
    _test(left_arg + C(left_arg) + right_arg, left_arg+expected)

def test_op_minus(left_arg, right_arg):
    expected = AA(left_arg) - AA(right_arg)
    _test(C(left_arg) - right_arg, expected)
    _test(C(left_arg) - C(right_arg), expected)
    _test(left_arg - C(right_arg), expected)
    _test(left_arg - C(left_arg) + right_arg, left_arg-expected)

def test_op_times(left_arg, right_arg):
    expected = AA(left_arg) * AA(right_arg)
    _test(C(left_arg) * right_arg, expected)
    _test(C(left_arg) * C(right_arg), expected)
    _test(left_arg * C(right_arg), expected)
