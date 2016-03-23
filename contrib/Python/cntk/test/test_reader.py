import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *
from .. import cntk1_ops

allclose = np.testing.assert_allclose

def test_NumPyReader(tmpdir):
    data = [[1, 2], [3, 4]]
    fn = str(tmpdir / 'test.txt')
    reader = NumPyReader(data, fn)

    input_node = cntk1_ops.Input(2, var_name='testInput')
    reader.add_input(input_node, 0,2)
    out = input_node + 2

    with get_new_context() as ctx:
        result = ctx.eval(out, reader)
        assert np.all(result == np.asarray(data)+2)

# TODO test other readers

