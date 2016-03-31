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

        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert np.allclose(res, exp)
            assert res.shape == AA(exp).shape

#C_VALUES = [0, [[1, 2], [3, 4]], [10.1, -20.2], 1.1]
C_VALUES = [0, [[1, 2], [3, 4]]]

@pytest.fixture(scope="module", params=C_VALUES)
def c_arg(request):
    return request.param

c_left_arg = c_arg
c_right_arg = c_arg

if False:
    def test_op_add_constant(c_left_arg, c_right_arg):
        expected = [AA(c_left_arg) + AA(c_right_arg)]
        _test(C(c_left_arg) + c_right_arg, expected, False)
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

# Testing inputs

@pytest.mark.parametrize("left_arg, right_arg", [
    ([30], [10]),
    ([[30]], [[10]]),
    ([[1.5,2.1]], [[10,20]]),
    # Adding two 3x2 inputs of sequence length 1
    ([[30,40], [1,2], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]), 
    ([5], [[30,40], [1,2]]),
    ])
def test_op_add_input_constant(left_arg, right_arg):
    expected = AA(left_arg) + AA(right_arg)
    # sequence of 1 element, since we have has_sequence_dimension=False
    expected = [expected] 
    # batch of one sample
    expected = [expected]
    _test(I([left_arg], has_sequence_dimension=False) + right_arg, expected, False)
    _test(left_arg + I([right_arg], has_sequence_dimension=False), expected, False)

@pytest.mark.parametrize("left_arg, right_arg", [
    ([
        [[30]],     # 1st element has (1,) sequence of length 1
        [[11],[12]] # 2nd element has (1,) sequence of length 2
     ] , 
      2), 
    ([
        [[33,22]],           # 1st element has (1x2) sequence of length 1
        [[11,12], [1.1,2.2]] # 2nd element has (1x2) sequence of length 2
     ],  
      2), 
    ])

def test_op_mul_input_seq(left_arg, right_arg):
    expected = [AA(elem)*right_arg for elem in left_arg]
    result = I(left_arg, has_sequence_dimension=True) * right_arg
    _test(result, expected, False)

