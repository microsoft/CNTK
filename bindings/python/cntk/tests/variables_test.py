# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for Variable and its descendents.
"""

from .. import times, placeholder, constant, plus, alias,\
    initializer
import numpy as np
import cntk as C
import pytest

VARIABLE_TYPES = [C.Constant, C.Parameter]


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

def test_parameter_set_value():
    p = C.Parameter(shape=(2,3), init=1);
    n = np.random.randn(2, 3)
    p.value = n
    assert np.all(p.value == n.astype(p.dtype))

    n = np.reshape(np.arange(6), (2, 3))
    p.value = n
    op = plus(p, p)
    state, output = op.forward({}, op.outputs, op.outputs)
    value = output[op.output]
    assert np.all(value == 2*n.astype(p.dtype))

    p.value = C.internal.sanitize_value(p.shape, 1.0, np.float32, None)
    assert np.all(p.value == np.ones((2,3)))

@pytest.mark.parametrize("value", VALUES)
def test_constant_value(value):
    c = C.Constant(value=value)
    assert np.allclose(c.value, value)

@pytest.mark.parametrize("value", VALUES)
def test_parameter_value(value):
    c = C.Parameter(init=value)
    assert np.allclose(c.value, value)

def test_constant():
    val = np.asarray([[1,2],[3,4]])
    c = constant(val)
    assert np.allclose(c.value, val)

def test_placeholder():
    p1 = placeholder(shape=(1,2))
    p2 = placeholder(shape=(2,1))
    op = times(p1, p2)

    p1 = placeholder(shape=(1,2))
    p2 = placeholder(shape=(2,1))
    op = times(p1, [[1],[2]])

def test_constant_shape_inf():
    shape = (-1,4)
    c = constant(value=2, shape=shape)
    assert np.allclose(c.shape, shape)
    with pytest.raises(ValueError):
        c.value

def test_convert_to_variable_dtype():
    assert C.input_variable(1).dtype == np.float32

    data = np.arange(1, dtype=np.int32)
    result = (C.input_variable(1)+2).eval([data])
    assert result==[[[2]]]
    assert result.dtype == np.float32

    data = np.arange(1., dtype=np.float64)
    (C.input_variable(1)+2).eval([data])
    result = (C.input_variable(1)+2).eval([data])
    assert result==[[[2]]]
    assert result.dtype == np.float32

def test_getitem():
    x = C.input_variable(5)
    y = x+0
    f = y[3]
    r = f.eval([np.array([[1, 2, 3, 4, 5]])])
    assert np.allclose(r, [[[4]]])
    f = y[2:4:1]
    r = f.eval([np.array([[1, 2, 3, 4, 5]])])
    assert np.allclose(r, [[[3,4]]])
    f = y[2::]
    r = f.eval([np.array([[1, 2, 3, 4, 5]])])
    assert np.allclose(r, [[[3,4,5]]])
    f = y[:3]
    r = f.eval([np.array([[1, 2, 3, 4, 5]])])
    assert np.allclose(r, [[[1,2,3]]])
    f = y[1:4:2]
    r = f.eval([np.array([[1, 2, 3, 4, 5]])])
    assert np.allclose(r, [[[2, 4]]])

def test_initializer_scale():
    # this should work fine:
    p = C.Parameter(shape=(1,), init=initializer.uniform(1));
    with pytest.raises(ValueError) as excinfo:
        name = 'uniform_zero'
        p = C.Parameter(shape=(1,), init=initializer.uniform(0), name=name);
        assert 'CreateInitializer' in str(excinfo.value)
        assert name in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        name = 'glorot_negative_one'
        p = C.Parameter(shape=(1,), init=initializer.glorot_uniform(-1), name=name);
        assert 'CreateInitializer' in str(excinfo.value)
        assert name in str(excinfo.value)