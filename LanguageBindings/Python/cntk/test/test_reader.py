import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *

allclose = np.testing.assert_allclose


def test_NumPyReader(tmpdir):
    data = [[1, 2], [3, 4]]
    fn = str(tmpdir / 'test.txt')
    reader = NumPyReader(data, fn)

    with Context('test', clean_up=False) as ctx:
        node = Input(2, ctx=ctx)
        result = ctx.eval(node, {node: (reader, (0, 2))})
        assert np.all(result == data)
