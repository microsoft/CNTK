# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for random number generation operations; forward only of course
"""

from __future__ import division
import numpy as np
import pytest
import cntk as C
import cntk.random as cr
from cntk.tests.test_utils import precision, PRECISION_TO_TYPE
from cntk.ops.tests.ops_test_utils import cntk_device

DIST_PARAMS = [
    (0.1, 1),
    (0.5, 1),
    (0.9, 1),
    (0.1, 2),
    (0.5, 2),
    (0.9, 2)
]


@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_randomlike_moments(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    x = C.input_variable(1, dtype=dt)
    N = 100000 
    B = 10.0 / np.sqrt(N)
    x0 = np.zeros((N, 1), dtype=dt)
    eg = np.euler_gamma

    #                  op             mean,                   variance
    ops1 = [(cr.bernoulli_like, lambda a: a           , lambda a: a*(1-a))]

    #                  op             mean,                   variance
    ops2 = [(cr.uniform_like,   lambda a, b: (b+a)*0.5, lambda a,b: (b-a)**2/12.0   ),
            (cr.normal_like,    lambda a, b: a        , lambda a,b: b**2            ),
            (cr.gumbel_like,    lambda a, b: a+b*eg   , lambda a,b: (np.pi*b)**2/6.0)]

    for op, fmean, fvar in ops1:
        input_op = op(x, arg0, seed=98052)
        value = input_op.eval({x: x0}, device=dev)
        assert np.abs(np.mean(value) - fmean(arg0)) < B
        assert np.abs(np.var(value) - fvar(arg0)) < B * fvar(arg0)

    for op, fmean, fvar in ops2:
        input_op = op(x, arg0, arg1, seed=98052)
        value = input_op.eval({x: x0}, device=dev)
        assert np.abs(np.mean(value) - fmean(arg0, arg1)) < B
        assert np.abs(np.var(value) - fvar(arg0, arg1)) < B * fvar(arg0,arg1)


@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_random_moments(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    N = 100000
    B = 10.0 / np.sqrt(N) # about 1.5 larger than the largest value ever observed
    eg = np.euler_gamma

    #                  op             mean,                   variance
    ops1 = [(cr.bernoulli, lambda a: a           , lambda a: a*(1-a))]

    #                  op             mean,                   variance
    ops2 = [(cr.uniform,   lambda a, b: (b+a)*0.5, lambda a,b: (b-a)**2/12.0   ),
            (cr.normal,    lambda a, b: a        , lambda a,b: b**2            ),
            (cr.gumbel,    lambda a, b: a+b*eg   , lambda a,b: (np.pi*b)**2/6.0)]

    for op, fmean, fvar in ops1:
        input_op = op((N//100,10,10), dt, arg0, seed=98052)
        value = input_op.eval(device=dev)
        assert np.abs(np.mean(value) - fmean(arg0)) < B
        assert np.abs(np.var(value) - fvar(arg0)) < B * fvar(arg0)

    for op, fmean, fvar in ops2:
        input_op = op((N//100,10,10), dt, arg0, arg1, seed=98052)
        value = input_op.eval(device=dev)
        assert np.abs(np.mean(value) - fmean(arg0, arg1)) < B
        assert np.abs(np.var(value) - fvar(arg0, arg1)) < B * fvar(arg0, arg1)


@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_two_times_n_vs_one_time_2n(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    N = 256
    x1 = np.zeros((N, 1), dtype=dt)
    x2 = np.zeros((2*N, 1), dtype=dt)

    ops1 = [cr.bernoulli]
    ops2 = [cr.uniform, cr.normal, cr.gumbel]

    for op in ops1:
        input_op1 = op((1*N,), dt, arg0, seed=98052)
        input_op2 = op((2*N,), dt, arg0, seed=98052)
        a = input_op1.eval(device=dev)
        b = input_op1.eval(device=dev)
        c = input_op2.eval(device=dev)
        assert np.allclose(c, np.concatenate([a,b]))
        
    for op in ops2:
        input_op1 = op((1*N,), dt, arg0, arg1, seed=98052)
        input_op2 = op((2*N,), dt, arg0, arg1, seed=98052)
        a = input_op1.eval(device=dev)
        b = input_op1.eval(device=dev)
        c = input_op2.eval(device=dev)
        assert np.allclose(c, np.concatenate([a,b]))


@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_normal_pair(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    N = 100000
    B = 10.0 / np.sqrt(N)
    input_op = cr.normal((N,), dt, arg0, arg1, seed=98052)
    a = input_op.eval(device=dev)
    b = input_op.eval(device=dev)
    value = a - b
    assert np.abs(np.mean(value)) < B
    assert np.abs(np.var(value) - 2*arg1*arg1) < np.sqrt(2)*arg1*B

@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_normal_diff_along_sequence(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    N = 100000
    B = 10.0 / np.sqrt(N)
    x  = C.sequence.input_variable(1, dtype=dt)
    x0 = np.zeros((2,N,1), dtype=dt)
    input_op = cr.normal_like(x, arg0, arg1, seed=98052)
    a = input_op.eval({x:x0}, device=dev)
    value = a[0] - a[1]
    assert np.abs(np.mean(value)) < B
    assert np.abs(np.var(value) - 2*arg1*arg1) < np.sqrt(2)*arg1*B

@pytest.mark.parametrize("arg0, arg1", DIST_PARAMS)
def test_normal_diff_along_batch(arg0, arg1, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    N = 1000
    B = 10.0 / np.sqrt(N)
    x  = C.sequence.input_variable(1, dtype=dt)
    x0 = np.zeros((N,2,1), dtype=dt)
    z = cr.normal_like(x, arg0, arg1, seed=98052)
    diff = C.sequence.first(z)-C.sequence.last(z)
    mean = C.reduce_mean(diff, axis=C.Axis.all_axes())
    var  = C.reduce_mean(diff*diff, axis=C.Axis.all_axes())
    expr = C.combine([mean, var])
    values = expr.eval({x:x0}, device=dev)
    assert np.abs(values[mean.output]) < B
    assert np.abs(values[var.output] - 2*arg1*arg1) < np.sqrt(2)*arg1*B


def test_placeholder(device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    import cntk.random as cr
    p = C.placeholder()
    u = cr.uniform_like(p)
    x = C.sequence.input_variable((4,5))

    x1 = np.ones((2,3,4,5), dtype=dt)
    f = u + p;
    f.replace_placeholders({p:x})
    fx0,fx1 = f.eval({x:x1})
    
    assert fx0.shape == (3,4,5)
    assert fx1.shape == (3,4,5)
    
    assert fx0.min() >= 1
    assert fx0.max() <  2

    assert fx1.min() >= 1
    assert fx1.max() <  2
    
