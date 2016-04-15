# Here should all the functional operator tests go.

import numpy as np
import pytest
from ..context import get_new_context
from ..graph import *
from ..reader import *

# keeping things short
C = constant
I = input_reader
AA = np.asarray

def _test(root_node, expected, clean_up=True, backward_pass = False, input_node = None):
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        assert not ctx.input_nodes
        result = ctx.eval(root_node, None, backward_pass, input_node)

        assert len(result) == len(expected)
        for res, exp in zip(result, expected):  
            assert np.allclose(res, exp)
            assert res.shape == AA(exp).shape

C_VALUES = [0, [[1, 2], [3, 4]]]

@pytest.fixture(scope="module", params=C_VALUES)
def c_arg(request):
    return request.param

c_left_arg = c_arg
c_right_arg = c_arg

#TODO: broken due to a problem in CNTK. Once fixed merge them with the tests in linear_test.py
if False:
    def test_op_add_constant(c_left_arg, c_right_arg):
        expected = [AA(c_left_arg) + AA(c_right_arg)]
        _test(C(c_left_arg) + c_right_arg, expected)
        _test(C(c_left_arg) + C(c_right_arg), expected)
        _test(c_left_arg + C(c_right_arg), expected)
        _test(c_left_arg + C(c_left_arg) + c_right_arg, c_left_arg+expected)

    def test_op_minus_constant(c_left_arg, c_right_arg):
        expected = [AA(c_left_arg) - AA(c_right_arg)]
        _test(C(c_left_arg) - c_right_arg, expected)
        _test(C(c_left_arg) - C(c_right_arg), expected)
        _test(c_left_arg - C(c_right_arg), expected)
        _test(c_left_arg - C(c_left_arg) + c_right_arg, c_left_arg-expected)

    def test_op_times_constant(c_left_arg, c_right_arg):
        expected = [AA(c_left_arg) * AA(c_right_arg)]
        _test(C(c_left_arg) * c_right_arg, expected)
        _test(C(c_left_arg) * C(c_right_arg), expected)
        _test(c_left_arg * C(c_right_arg), expected)
