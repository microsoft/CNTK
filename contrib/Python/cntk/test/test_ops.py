# Here should all the functional operator tests go.

import numpy as np
import pytest
from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..reader import *

# keeping things short
C = constant
I = input


def _test(root_node, expected, clean_up=True):
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        assert not ctx.input_nodes
        result = ctx.eval(root_node)
        expected = np.asarray(expected)
        assert result.shape == expected.shape or result.shape == (
            1, 1) and expected.shape == ()
        assert np.all(result == expected)


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

    (C(np.ones((2, 3)) * 3), [[3, 3, 3], [3, 3, 3]]),
    (C(np.ones((2, 3)) * 3) + \
     np.vstack([np.ones(3), np.ones(3) + 1]), [[4, 4, 4], [5, 5, 5]]),
    (C(np.ones((2, 3)) * 3) * \
     np.vstack([np.ones(3), np.ones(3) + 1]), [[3, 3, 3], [6, 6, 6]]),

    # special treatment of inputs in RowStack
    # (RowStack((C(1), C(2))), [[1],[2]]), # TODO figure out the real semantic
    # of RowStack

    # the following test fails because Constant() ignores the cols parameter
    #(RowStack((C(1, rows=2, cols=2), C(2, rows=2, cols=2))), [[1,1,2,2], [1,1,2,2]])

    # __abs__
    # uncomennt, once Abs() as ComputationNode is moved from standard function
    # to ComputationNode
    (abs(C(-3)), 3),
    (abs(C(3)), 3),
    (abs(C([[-1, 2], [50, -0]])), [[1, 2], [50, 0]]),

    # more complex stuff
    #(Plus(C(5), 3), 8),
])
def test_overload_eval(root_node, expected):
    _test(root_node, expected)


@pytest.mark.parametrize('root_node, expected', [
    # __add__ / __radd__
    (C(np.asarray([1, 2])) + 0, [1, 2]),
    (C(np.asarray([1, 2])) + .1, [1.1, 2.1]),
    (.1 + C(np.asarray([1, 2])), [1.1, 2.1]),
    (C(np.asarray([1, 2])) * 0, [0, 0]),
    (C(np.asarray([1, 2])) * .1, [0.1, 0.2]),
    (.1 * C(np.asarray([1, 2])), [0.1, 0.2]),
    (C(np.asarray([[1, 2], [3, 4]])) + .1, [[1.1, 2.1], [3.1, 4.1]]),
    (C(np.asarray([[1, 2], [3, 4]])) * 2, [[2, 4], [6, 8]]),
    (2 * C(np.asarray([[1, 2], [3, 4]])), [[2, 4], [6, 8]]),
    (2 * C(np.asarray([[1, 2], [3, 4]])) + 100, [[102, 104], [106, 108]]),
    (C(np.asarray([[1, 2], [3, 4]]))
     * C(np.asarray([[1, 2], [3, 4]])), [[1, 4], [9, 16]]),
])
def test_ops_on_numpy(root_node, expected, tmpdir):
    _test(root_node, expected, clean_up=False)
