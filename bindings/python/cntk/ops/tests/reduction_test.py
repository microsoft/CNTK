# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for reduction operations, tested for the forward and the backward pass
"""

from __future__ import division
import numpy as np
import itertools
import pytest
import cntk as C
from .ops_test_utils import unittest_helper, _test_unary_op, AA, precision, PRECISION_TO_TYPE, constant
from cntk.internal import sanitize_dtype_cntk
from .. import reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_log_sum_exp, reduce_prod

REDUCE_TEST_OPERANDS = [
    #(input_data,  axis)
    ([[1]], 0),
    ([[1]], [0]),
    ([[1]], [0,1]),
    ([[1,2],[4,5]], 0),
    ([[1, 2], [4, 5]], [0]),
    ([[1, 2], [4, 5]], [0,1]),
    ([[1,2],[4,5]], 1),
    ([[1,2],[4,5]], -1),
    ([[1, 2], [4, 5]], [-1]),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], -2),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [0, -2]),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], 2),
    ([[[1,2],[3,4]],[[5,6],[7,8]]], [1,2]),
]

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_sum(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    axis = tuple(axis) if isinstance(axis, list) else (axis)
    expected_forward = [np.sum(data, axis=axis, keepdims=True)]
    backward = np.ones_like(data)

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_sum
    _test_unary_op(precision, device_id, reduce_sum, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_max(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)

    axis = tuple(axis) if isinstance(axis, list) else (axis)
    expected_forward = [np.amax(data, axis=axis, keepdims=True)]

    forward_array = np.asarray(expected_forward, dtype=dt)
    max_elements = forward_array.reshape(forward_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(data)
    for element in max_elements:
        backward += np.asarray(data == element)

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_max
    _test_unary_op(precision, device_id, reduce_max, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_min(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)
    axis = tuple(axis) if isinstance(axis, list) else (axis)

    expected_forward = [np.amin(data, axis=axis, keepdims=True)]

    forward_array = np.asarray(expected_forward, dtype=dt)
    max_elements = forward_array.reshape(forward_array.size).tolist()

    # place 1.0s where maximum elements are
    backward = np.zeros_like(data)
    for element in max_elements:
        backward += np.asarray(data == element)

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_min
    _test_unary_op(precision, device_id, reduce_min, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_mean(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)
    axis = tuple(axis) if isinstance(axis, list) else (axis)
    expected_forward = [np.mean(data, axis=axis, keepdims=True)]

    counts = np.prod([data.shape[a] for a in axis]) if type(axis) in [list, tuple] else data.shape[axis]
    backward = np.ones_like(data) / counts

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_mean
    _test_unary_op(precision, device_id, reduce_mean, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_log_sum(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)
    axis = tuple(axis) if isinstance(axis, list) else (axis)

    data_exp = np.exp(data)
    sum_exp = np.sum(data_exp, axis=axis, keepdims=True)
    expected_forward = [np.log(sum_exp)]

    backward = data_exp / sum_exp

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_log_sum_exp
    _test_unary_op(precision, device_id, reduce_log_sum_exp, input_data,
                   expected_forward, expected_backward, {'axis': axis})

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_prod(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)
    axis = tuple(axis) if isinstance(axis, list) else (axis)

    p = np.prod(data, axis=axis, keepdims=True)
    expected_forward = [p]

    backward = p / data

    expected_backward = {
        'arg': [backward]
    }

    from .. import reduce_prod
    _test_unary_op(precision, device_id, reduce_prod, input_data,
                   expected_forward, expected_backward, {'axis': axis})
                   
@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_all(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)
    a = C.sequence.input_variable(shape=data.shape,
                         dtype=sanitize_dtype_cntk(dt),
                         needs_gradient=True,
                         name='a')
    # create batch
    value = [AA([data,data-0.5], dtype=dt),AA([data+0.25], dtype=dt)]
    from .. import reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_log_sum_exp, reduce_prod
    from cntk import Axis
    def max_bwd(x,f):
        y = np.zeros_like(x)
        yr = y.ravel()
        xr = x.ravel()
        for i in range(x.size):
            if xr[i] == f: yr[i] = 1
        return y

    ops = [ (reduce_sum,         lambda x:AA(sum(np.sum(xi) for xi in x)),                           lambda x,f:[np.ones_like(xi) for xi in x]),
            (reduce_max,         lambda x:AA(max(np.max(xi) for xi in x)),                           lambda x,f:[max_bwd(xi,f) for xi in x]),
            (reduce_min,         lambda x:AA(min(np.min(xi) for xi in x)),                           lambda x,f:[max_bwd(xi,f) for xi in x]),
            (reduce_mean,        lambda x:AA(sum(np.sum(xi) for xi in x)/sum(xi.size  for xi in x)), lambda x,f:[np.ones_like(xi)/sum(xj.size for xj in x) for xi in x]),
            (reduce_log_sum_exp, lambda x:AA(np.log(sum(np.sum(np.exp(xi)) for xi in x))),           lambda x,f:[np.exp(xi-f)     for xi in x]),
            (reduce_prod,        lambda x:AA(np.prod([np.prod(xi) for xi in x])),                    lambda x,f:[f/xi             for xi in x])
            ]

    for op,fwd,bwd in ops:
        input_op = op(a, axis=Axis.all_axes())
        expected_forward = fwd(value)
        expected_backward = bwd(value,expected_forward)
        binding = {a: value}
        actual_backward = input_op.grad(binding)
        actual_forward  = input_op.eval(binding)
        assert np.allclose(actual_forward, expected_forward)
        for ab,eb in zip (actual_backward, expected_backward):
            assert np.allclose(ab, eb)

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_mean_all_constant(input_data, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]
    value = AA(input_data, dtype=dt)
    from .. import reduce_mean
    from cntk import Axis, Constant
    a = Constant(value, name='a')
    input_op = reduce_mean(a, axis=Axis.all_axes())
    expected_forward = AA(np.mean(value))
    actual_forward  = input_op.eval()
    assert np.allclose(actual_forward, expected_forward)

REDUCE_BATCH_TEST_OPERANDS = [
    #(input_data)
    ([[1]]),
    ([[1,2],[4,5]]),
    ([[[1,2],[3,4]],[[5,6],[7,8]]]),
]


#TODO: The implementation of this functon requires that all values in the data are distinct.
def min_max_bwd(x, f, dtype):
    forward_array = np.asarray(f, dtype=dtype)
    min_max_elements = forward_array.reshape(forward_array.size).tolist()
    # place 1.0s where minimum or maximum elements are
    backward = np.zeros_like(x)
    for element in min_max_elements:
        backward += np.asarray(x == element)
    return backward


@pytest.mark.parametrize("input_data", REDUCE_BATCH_TEST_OPERANDS)
def test_op_reduce_over_batch_axis(input_data, device_id, precision):
    from .. import reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_log_sum_exp, reduce_prod
    from cntk import Axis

    dt = PRECISION_TO_TYPE[precision]

    data = AA(input_data, dtype=dt)
    a = C.input_variable(shape=data.shape[1:],
                dtype=sanitize_dtype_cntk(dt),
                needs_gradient=True,
                name='a')
    ops = [
            (reduce_sum,         lambda x:np.sum(x, axis=0, keepdims=False),                    lambda x,f:np.ones_like(x)),
            (reduce_max,         lambda x:np.amax(x, axis=0, keepdims=False),                   lambda x,f:min_max_bwd(x,f, dt)),
            (reduce_min,         lambda x:np.amin(x, axis=0, keepdims=False),                   lambda x,f:min_max_bwd(x,f, dt)),
            (reduce_mean,        lambda x:np.mean(x, axis=0, keepdims=False),                   lambda x,f:np.ones_like(x)/x.shape[0]),
            (reduce_log_sum_exp, lambda x:np.log(np.sum(np.exp(x), axis=0, keepdims=False)),    lambda x,f:np.exp(x-f)),
            (reduce_prod,        lambda x:np.prod(x, axis=0, keepdims=False),                   lambda x,f:f / x)
          ] 

    for op,fwd,bwd in ops:
        input_op = op(a, axis=Axis.default_batch_axis())
        expected_forward = fwd(data)
        expected_backward = bwd(data, expected_forward)
        binding = {a: data}
        actual_backward = input_op.grad(binding)
        actual_forward  = input_op.eval(binding)
        assert np.allclose(actual_forward, expected_forward)
        for ab,eb in zip (actual_backward, expected_backward):
            assert np.allclose(ab, eb)

@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_argmax(input_data, axis, device_id, precision):
    if isinstance(axis, list):
        #following numpy, argmax over multiple axis is not supported
        return
    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    # numpy argmax doesn't support keepdims
    arg_shape = np.amax(data, axis=axis, keepdims=True).shape
    expected_forward = [np.argmax(data, axis=axis).reshape(arg_shape)]

    from .. import argmax
    _test_unary_op(precision, device_id, argmax, input_data,
                   expected_forward, None, {'axis': axis})


@pytest.mark.parametrize("input_data, axis", REDUCE_TEST_OPERANDS)
def test_op_reduce_argmin(input_data, axis, device_id, precision):
    if isinstance(axis, list):
        #following numpy, argmin over multiple axis is not supported
        return;

    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)

    # numpy argmin doesn't support keepdims
    arg_shape = np.amin(data, axis=axis, keepdims=True).shape
    expected_forward = [np.argmin(data, axis=axis).reshape(arg_shape)]

    from .. import argmin
    _test_unary_op(precision, device_id, argmin, input_data,
                   expected_forward, None, {'axis': axis})

#Note that due to the limited capability of np.amax (there is no
#way to figure out which multi-dimensional index corresponds to
#the max or min values. To test the backward gradicent of
#reduce_max and reduce_min, we have to enforce all the
#numbers in the data set must be distinct to identify the index of
#max and min along multi-dimensional axes.
REDUCE_BATCH_SEQUENCE_STATIC_TEST_DATA = [
    [#a data set
    # batch axis:
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], ],  # a sequence
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]], ],  # a sequence
    ],
    [  # a data set
        # batch axis:
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], ],  # a sequence
        [[[17, 18], [19, 20]], [[21, 22], [23, 24]], ],  # a sequence
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]], ],  # a sequence
    ],
]

REDUCE_BATCH_SEQUENCE_STATIC_TEST_AXES =[
    [],
    [0],
    [1],
    [0,1]
]

REDUCE_BATCH_SEQUENCE_DYNAMIC_TEST_AXES =[
    #test reduction over batch axis + static axes;
    #when static axes = [], test reduction over batch axis
    [C.Axis.default_batch_axis()],
    #test reduction over sequence axis + static axes;
    #when static axes = [], test reduction over sequence axis
    #TODO: Comment the following out before we can have an agreement on whether we allow sequence axis here for now
    #[C.Axis.default_dynamic_axis()],
    #test reduction over batch axis + sequence axis  + static axes;
    #when static axes = [], test reduction over batch axis + sequence axis
    #[C.Axis.default_batch_axis(), C.Axis.default_dynamic_axis()],
]

REDUCE_BATCH_SEQUENCE_STATIC_TEST_OPERANDS = itertools.product(REDUCE_BATCH_SEQUENCE_STATIC_TEST_DATA, REDUCE_BATCH_SEQUENCE_DYNAMIC_TEST_AXES, REDUCE_BATCH_SEQUENCE_STATIC_TEST_AXES)

def reduce_batch_sequence_static_ops(dtype):
    return [
    (reduce_sum, lambda x, axes: np.sum(x, axis=axes, keepdims=True), lambda x, f, axes: np.ones_like(x)),
    (reduce_max, lambda x, axes: np.amax(x, axis=axes, keepdims=True), lambda x, f, axes: min_max_bwd(x, f, dtype)),
    (reduce_min, lambda x, axes: np.amin(x, axis=axes, keepdims=True), lambda x, f, axes: min_max_bwd(x, f, dtype)),
    (reduce_mean, lambda x, axes: np.mean(x, axis=axes, keepdims=True),
     lambda x, f, axes: np.ones_like(x) / np.prod([x.shape[a] for a in axes])),
    (reduce_log_sum_exp, lambda x, axes: np.log(np.sum(np.exp(x), axis=axes, keepdims=True)),
     lambda x, f, axes: np.exp(x - f)),
    (reduce_prod, lambda x, axes: np.prod(x, axis=axes, keepdims=True), lambda x, f, axes: f / x)
    ]

#aux function to the ops
def _test_reduce_ops(input_var, data, op, cntk_axes, numpy_axes, fwd, bwd):
    expected_forward = fwd(data, numpy_axes)
    expected_backward = bwd(data, expected_forward, numpy_axes)
    binding = {input_var: data}

    cntk_op = op(input_var, axis=cntk_axes)
    actual_backward = cntk_op.grad(binding)
    actual_forward = cntk_op.eval(binding)

    assert np.allclose(actual_forward, expected_forward)
    for ab, eb in zip(actual_backward, expected_backward):
        assert np.allclose(ab, eb)


@pytest.mark.parametrize("input_data, dynamic_axes, static_axes", REDUCE_BATCH_SEQUENCE_STATIC_TEST_OPERANDS)
def test_op_reduce_batch_sequence_static_axes_together(input_data, dynamic_axes, static_axes, device_id, precision):
    from cntk import Axis

    dt = PRECISION_TO_TYPE[precision]
    data = AA(input_data, dtype=dt)
    if dynamic_axes == [C.Axis.default_batch_axis()]:
        #Reduction along the batch axis on input sequence is currently unsupported, so only batch axis input is tested
        v = C.input_variable(data.shape[1:],
                                  dtype=sanitize_dtype_cntk(dt),
                                  needs_gradient=True)
        numpy_axis_offset = 1
        ignore_max_min = True
    else:
        v = C.sequence.input_variable(data.shape[2:],
                                  dtype=sanitize_dtype_cntk(dt),
                                  needs_gradient=True)
        numpy_axis_offset = 2
        ignore_max_min = False


    for op, fwd, bwd in reduce_batch_sequence_static_ops(dt):
        cntk_axes = tuple(dynamic_axes + static_axes)
        numpy_axes = tuple([0 if a == C.Axis.default_batch_axis() else 1 for a in dynamic_axes] + [ax + numpy_axis_offset for ax in static_axes])
        _test_reduce_ops(v, data, op, cntk_axes, numpy_axes, fwd, bwd)
