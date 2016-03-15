from ..graph import *

import pytest

# keeping things short
C = Constant


# testing whether operator overloads result in proper type
@pytest.mark.parametrize('root_node, expected', [
    # __add__ / __radd__
    (C(0) + C(1), Plus),
    (C(0) + 1, Plus),
    (0 + C(1), Plus),
    (0 + 1, int),

    # __sub__ / __rsub__
    (C(0) - C(1), Minus),
    (C(0) - 1, Minus),
    (0 - C(1), Minus),
    (0 - 1, int),

    # __mul__ / __rmul__ --> element-wise (!) multiplication
    (C(0) * C(1), ElementTimes),
    (C(0) * 1, ElementTimes),
    (0 * C(1), ElementTimes),
    (0 * 1, int),

    # __abs__
    (abs(C(0)), Abs),

    # __getitem__
    (C(range(0, 10))[2:5], RowSlice),
    (C(range(0, 10))[:5], RowSlice),

])
def test_overload_types(root_node, expected):
    assert isinstance(root_node, expected)


def test_overload_exception():
    with pytest.raises(ValueError):
        C(range(0, 10))[:]

    with pytest.raises(ValueError):
        C(range(0, 10))[0:3:2]


@pytest.mark.parametrize("root_node, expected", [
    (Input(2), "v0 = Input(2, tag='feature')"),
    (Plus(C(0), C(1)),
     "v0 = Constant(0, rows=1, cols=1)\nv1 = Constant(1, rows=1, cols=1)\nv2 = Plus(v0, v1)"),
])
def test_description(root_node, expected):
    description, has_inputs = root_node.to_description() 
    assert description == expected


def test_graph_with_same_node_twice():
    v0 = C(1)
    root_node = Plus(v0, v0)
    expected = 'v0 = Constant(1, rows=1, cols=1)\nv1 = Plus(v0, v0)'
    description, has_inputs = root_node.to_description() 
    assert description == expected
