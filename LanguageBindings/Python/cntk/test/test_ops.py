# Here should all the functional operator tests go.

import numpy as np
import pytest
from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..reader import *

# keeping things short
C = Constant

@pytest.mark.parametrize('root_node, expected', [
    # __add__ / __radd__
    (C(0) + C(1), 1),
    (C(0) + 1, 1),
    (0 + C(1), 1),

    # __sub__ / __rsub__
    (C(0) - C(1), -1),
    (C(0) - 1, -1),
    (0 - C(1), -1),

    # __mul__ / __rmul__ --> element-wise (!) multiplication
    (C(0) * C(1), 0),
    (C(0) * 1, 0),
    (0 * C(1), 0),

    # chaining
    (C(2) * C(3) + C(1.2), 7.2),
    (C(2) * (C(3) + C(1.2)), 8.4),

    # normal ops

    #(Reshape(C(1, rows=4), numRows=2), [[1,1], [1,1]]),

    # wrapping it in a Plus, since Constants cannot be output
    #(C(3, 2, 3)+0, [[3,3,3], [3,3,3]]),

    # special treatment of inputs in RowStack 
    (RowStack((C(1), C(2))), [1,2]),
    #(RowStack((C(1, rows=2, cols=2), C(2, rows=2, cols=2))), [[1,1,2,2], [1,1,2,2]])

    # __abs__
    # uncomennt, once Abs() as ComputationNode is moved from standard function
    # to ComputationNode
    #(abs(C(-3)), 3),
    #(abs(C(3)), 3),

    # more complex stuff
    #(Plus(C(5), 3), 8),
])
def test_overload_eval(root_node, expected, tmpdir):
    with get_new_context() as ctx:
        #ctx.clean_up = False
        assert not ctx.input_nodes
        result = ctx.eval(root_node)
        expected = np.asarray(expected)
        assert result.shape == expected.shape
        assert np.all(result == expected)
