# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *
from ..ops import cntk1 as cntk1_ops
from ..ops import constant, input_numpy, dynamic_axis

from cntk.tests.test_utils import *

# Keeping things short
C = constant
I = input_numpy


def test_two_inputs(device_id, precision):
    a = AA([[1, 2]])
    b = AA([[10, 20]])

    expected = a + b

    axis = dynamic_axis()
    op_node = I([a], dynamic_axis=axis) + \
        I([b], dynamic_axis=axis)

    unittest_helper(op_node, None, [expected], device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)


def test_serialize_unmapped_node(tmpdir):
    tmpfile = str(tmpdir / 'out.txt')
    from cntk.reader import LazyInputReader
    axis = dynamic_axis()
    i1 = input_numpy(
        # 2 samples with 2 sequences each
        [
            AA([[[1, 2]], [[3, 4]]]),
            AA([[[10, 20]]])
        ], alias='X', dynamic_axis=axis)

    i2 = input_numpy(
        # 2 samples with 1 sequence each
        [
            AA([[[44, 55]]]),
            AA([[[66, 77]]])
        ], dynamic_axis=axis)

    expected = '''\
0	|X 1 2 |_I_0 44 55
0	|X 3 4
1	|X 10 20 |_I_0 66 77
'''

    im = InputMap()
    im._add_unmapped(i1)
    im._add_unmapped(i2)
    im._serialize_unmapped_nodes(tmpfile)

    with open(tmpfile, 'r') as f:
        assert f.read() == expected
