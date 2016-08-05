# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from ..graph import *
#from ..ops.cntk2 import Abs, Plus, Minus, ElementTimes
from ..ops import constant, input, plus, times, past_value
import numpy as np

import pytest

# keeping things short
A = np.asarray
C = constant
I = input


def test_overload_exception():
    c = C(list(range(0, 10)))

    with pytest.raises(TypeError):
        c[:]

    with pytest.raises(TypeError):
        c[0:3:2]


