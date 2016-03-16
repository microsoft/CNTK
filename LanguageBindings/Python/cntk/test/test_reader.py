import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *

allclose = np.testing.assert_allclose

def test_NumPyReader(tmpdir):
    data = [[1,2], [3,4]]
    fn = str(tmpdir / 'test.txt')
    reader = NumPyReader(data, fn)

    with get_new_context() as ctx:
        node = Input(2)
        result = ctx.eval(node, {node:(reader, (0,2))})
        assert np.all(result == data)

def test_NumPyReader_more_nodes(tmpdir):
    data = [[1, 2], [3, 4]]
    fn = str(tmpdir / 'test.txt')
    reader = NumPyReader(data, fn)

    input_node = Input(2)
    out = input_node + 2

    with get_new_context() as ctx:
        result = ctx.eval(out, {input_node: (reader, (0, 2))})
        assert np.all(result == np.asarray(data)+2)

# TODO test NumPyReader more extensively
# TODO test other readers

