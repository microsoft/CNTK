# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for Variable and its descendents.
"""

from ..variables import *
from .. import times, placeholder_variable, constant
import numpy as np

import pytest

VARIABLE_TYPES = [Constant, Parameter]


@pytest.mark.parametrize("variable_type", VARIABLE_TYPES)
def test_dtype(variable_type):
    c = variable_type(shape=(2,3))
    assert c.dtype == np.float32

    c = variable_type(shape=(2,3), dtype=np.float32)
    assert c.dtype == np.float32

    c = variable_type(shape=(2,3), dtype=np.float64)
    assert c.dtype == np.float64

@pytest.mark.parametrize("variable_type", VARIABLE_TYPES)
@pytest.mark.parametrize("shape", [(), 1, (1,), (1,2,3)])
def test_variable_shape(variable_type, shape):
    c = variable_type(shape=shape)
    if not isinstance(shape, tuple):
        shape = (shape,)
    assert c.shape == shape, variable_type

VALUES = [
        [1], 
        [[1],[2]], 
        [[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]
        ]

@pytest.mark.parametrize("value", VALUES)
def test_constant_value(value):
    c = Constant(value=value)
    assert np.allclose(c.value, value)

@pytest.mark.parametrize("value", VALUES)
def test_parameter_value(value):
    c = Parameter(init=value)
    assert np.allclose(c.value, value)

def test_constant():
    val = np.asarray([[1,2],[3,4]])
    c = constant(val)
    assert np.allclose(c.value, val)

def test_placeholder():
    p1 = placeholder_variable(shape=(1,2))
    p2 = placeholder_variable(shape=(2,1))
    op = times(p1, p2)

    p1 = placeholder_variable(shape=(1,2))
    p2 = placeholder_variable(shape=(2,1))
    op = times(p1, [[1],[2]])

def test_constant_shape_inf():
    shape = (-1,4)
    c = constant(value=2, shape=shape)
    assert np.allclose(c.shape, shape)
    with pytest.raises(ValueError):
        c.value
