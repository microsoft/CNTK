# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..context import get_new_context, _CONTEXT
from ..graph import *
#from ..ops.cntk2 import Abs, Plus, Minus, ElementTimes
from ..ops import constant, input, plus, times, past_value
import numpy as np

import pytest

# keeping things short
A = np.asarray
C = constant
I = input


'''
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
])
def test_overload_types(root_node, expected):
    assert isinstance(root_node, expected)
    '''


def test_overload_exception():
    c = C(list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
        c[0:3:2]


