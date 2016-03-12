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
    (C(0)+C(1), 1),
    (C(0)+1, 1),
    (0+C(1), 1),

    # __sub__ / __rsub__
    (C(0)-C(1), -1),
    (C(0)-1, -1),
    (0-C(1), -1),

    # __mul__ / __rmul__ --> element-wise (!) multiplication
    (C(0)*C(1), 0),
    (C(0)*1, 0),
    (0*C(1), 0),

    # chaining
    (C(2)*C(3)+C(1.2), 7.2),
    (C(2)*(C(3)+C(1.2)), 8.4),

    # __abs__
    #(abs(C(-3)), 3),
    #(abs(C(3)), 3),

    # more complex stuff
    #(Plus(C(5), 3), 8),
    ])
def test_overload_eval(root_node, expected, tmpdir):
    with get_new_context() as ctx:
        assert not ctx.graph.input_nodes 
        ctx.clean_up = False
        result = ctx.eval(root_node, {})
        assert np.all(result == expected)

    
