# Here should all the functional operator tests go.

import pytest
from ..context import get_context
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
    (C(0)*C(1), 1),
    (C(0)*1, 1),
    (0*C(1), 1),

    # __abs__
    (abs(C(-3)), 3),
    (abs(C(3)), 3),

    ])
def test_overload_eval(root_node, expected, tmpdir):
    with get_context() as ctx:
        ctx.clean_up = False
        result = ctx.eval(root_node, None)
        assert np.all(result == data)

    
