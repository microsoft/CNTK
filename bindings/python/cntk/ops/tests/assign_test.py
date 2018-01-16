# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the assign operation.
"""

import numpy as np
import pytest
import cntk as C
from .ops_test_utils import unittest_helper, _test_unary_op, AA, precision, PRECISION_TO_TYPE, constant
from cntk.internal import sanitize_dtype_cntk

ASSIGN_TEST_OPERANDS = [
    #(input_data)
    ([[1]]),
    ([[1,2],[4,5]]),
    ([[[1,2],[3,4]],[[5,6],[7,8]]]),
    ([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]]),
]

@pytest.mark.parametrize("input_data", ASSIGN_TEST_OPERANDS)
def test_assign_to_constant(input_data, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    value = C.parameter(init=data)
    dest = C.constant(shape=data.shape, dtype=dt)
    assign_op = C.assign(dest, value)

    result = assign_op.eval()

    assert np.array_equal(dest.asarray(), data)
    assert np.array_equal(result, data)

@pytest.mark.parametrize("input_data", ASSIGN_TEST_OPERANDS)
def test_assign_to_param(input_data, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    value = C.parameter(init=data)
    dest = C.parameter(shape=data.shape, dtype=dt)
    assign_op = C.assign(dest, value)

    result = assign_op.eval()

    assert np.array_equal(dest.asarray(), data)
    assert np.array_equal(result, data)

@pytest.mark.parametrize("input_data", ASSIGN_TEST_OPERANDS)
def test_assign_dependency(input_data, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    value = C.parameter(init=data)
    dest = C.parameter(shape=data.shape, dtype=dt)
    assign_op = C.assign(dest, value)
    y = dest + value

    result = C.combine([y, assign_op]).eval()

    assert np.array_equal(result[y.output], data)
    assert np.array_equal(dest.asarray(), data)
    assert np.array_equal(y.eval(), data + data)

@pytest.mark.parametrize("input_data", ASSIGN_TEST_OPERANDS)
def test_assign_gradient(input_data, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    value = C.parameter(init=data)
    dest = C.parameter(shape=data.shape, dtype=dt)
    assign_op = C.assign(dest, value)

    bwd, fwd = assign_op.forward({}, [assign_op.output], set([assign_op.output]))
    # dest will be updated after backprop, so verify that it isn't updated yet.
    assert not np.array_equal(dest.asarray(), data)

    result = list(fwd.values())[0]
    grad = assign_op.backward(bwd, {assign_op.output:np.ones_like(result)}, set([dest]))

    # dest should have the new value by now.
    assert np.array_equal(dest.asarray(), data)

    # check the gradient.
    assert np.array_equal(grad[dest], np.zeros_like(result))

@pytest.mark.parametrize("input_data", ASSIGN_TEST_OPERANDS)
def test_assign_timestamp_bump(input_data, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    x = C.parameter(shape=data.shape, dtype=dt)
    w = C.constant(np.ones_like(data))
    y = w + 1
    z = C.assign(x, y)

    f = x + 1
    result = f.eval()
    assert np.array_equal(x.asarray(), 0*np.ones_like(data))
    assert np.array_equal(result, 1*np.ones_like(data))

    result = z.eval()
    assert np.array_equal(x.asarray(), 2*np.ones_like(data))
    assert np.array_equal(result, 2*np.ones_like(data))

    result = f.eval()
    assert np.array_equal(x.asarray(), 2*np.ones_like(data))
    assert np.array_equal(result, 3*np.ones_like(data))



