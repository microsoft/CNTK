import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *

def test_NumPyReader(tmpdir):
    data = [[1,2], [3,4]]
    fn = str(tmpdir / 'test.txt')
    reader = NumPyReader(data, fn)

    with Context('test') as ctx:
        graph = Input(2)
        result = ctx.eval(graph, reader)
        assert result == np.asarray(data)


