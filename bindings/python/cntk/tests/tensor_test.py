# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..tensor import *
#from ..ops.cntk2 import Abs, Plus, Minus, ElementTimes
from ..ops import constant, input_variable, plus, times, past_value
import numpy as np

import pytest

# keeping things short
A = np.asarray
I = input_variable


def test_overload_exception():
    c = constant(value=list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
        c[0:3:2]


def test_eval_scalar():
    c = constant(value=2)
    assert (c+3).eval() == 5
    assert np.all((c+[3,4]).eval() == [5,6])
