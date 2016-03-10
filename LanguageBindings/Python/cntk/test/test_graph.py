from ..graph import *

import pytest

def test_add_overload():
    n1 = ComputationNode("A")
    n2 = ComputationNode("B")
    s = n1+n2
    assert isinstance(s, Plus)
    assert s.params[0] == 'leftMatrix'
    assert s.params[1] == 'rightMatrix'

@pytest.mark.parametrize("root_node, expected", [
    (Input(2), "v0 = Input(dims=2, tag='feature')"),
    (Label(2), "v0 = Input(dims=2, tag='label')"),
    (Plus(Constant(0), Constant(1)), "v0 = Constant(value=0, rows=1, cols=1)\nv1 = Constant(value=1, rows=1, cols=1)\nv2 = Plus(v0, v1)"),
    ])
def test_description(root_node, expected):
    assert root_node.to_description() == expected

def test_graph_with_same_node_twice():
    v0 = Constant(1)
    root_node = Plus(v0, v0)
    expected = 'v0 = Constant(value=1, rows=1, cols=1)\nv1 = Plus(v0, v0)'
    assert root_node.to_description() == expected

