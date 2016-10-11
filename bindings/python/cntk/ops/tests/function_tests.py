# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the function class.
"""

import numpy as np
import pytest
from ..functions import *
from ...trainer import *
from .. import constant, parameter, input_variable, placeholder_variable


def test_variable_forwarding():
    op = constant(value=2, shape=(3,4)) + 1
    assert op.shape().dimensions() == (3,4)


def test_replace_placeholders():
    p = placeholder_variable(shape=(1,))
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='a')
    res = p + 3
    res.replace_placeholders({p: i})

    assert res.eval({i: [3]}) == [6]

    res2 = p + 2
    from .. import plus
    func = plus(res2, 10)
    res2.replace_placeholders({p: func})

    assert res2.eval({i: [3]}) == [15]

