# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operations. Each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, cntk_device
from cntk.tests.test_utils import TOLERANCE_ABSOLUTE
from cntk.utils import eval as cntk_eval, sanitize_dtype_cntk
from .. import constant
from ..variables import Parameter, Constant
from cntk import set_default_device

EPS_IN_LOG = 1e-37        # 1e-37 is the highest guaranteed precision
# the backward result returned by CNTK log() for epsilon
BACKWARD_RESULST_FOR_LOG_EPS = 9.08782e+36
LOG_OF_EPS_IN_LOG = -85.1  # log(EPS_IN_LOG)

CLIP_TUPLES = [
    ([1.0], [2.0], [1.5]),  # value shouldn't be clipped; gradient is [1.0]
    ([1.0], [2.0], [0.5]),  # value should be clipped to 1.0; gradient is [0.0]
    ([1.0], [2.0], [2.5]),  # value should be clipped to 2.0; gradient is [0.0]

    # should clip to [1.5, 2.0, 1.0]; gradient is [[1.0, 0.0, 0.0]]
    ([1.0], [2.0], [[1.5, 2.1, 0.9]]),

    # should clip to [[1.0, 2.0], [1.0, 2.0], [1.5, 2.0]];
    # gradient is [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ([1.0], [2.0], [[0.0, 3.0], [1.0, 2.0], [1.5, 2.5]]),

    # test what happens if a user puts a higher "min" value than their "max" value
    # should clip to [[5.0, 5.0, 5.0, 5.0, 5.0]] because min is evaluated first
    # gradient should be all zeros: [[0.0, 0.0, 0.0, 0.0, 0.0]]
    ([5.0], [0.5], [[1.5, 2.1, 0.9, -1.0, -2.0]]),

    # test a more complicated broadcasting scenario
    ([[1.5, 2.0], [2.5, 3.0]], [[-2.0, 2.5], [2.5, 3.5]], [[-1.0, 2.0], [3.0, 4.0]]),
]


@pytest.mark.parametrize("min_value, max_value, x", CLIP_TUPLES)
def test_op_clip(min_value, max_value, x, device_id, precision):
    from .. import clip
    dev = cntk_device(device_id)

    expected_forward = [np.clip(AA([x], dtype=PRECISION_TO_TYPE[precision]), AA(
        min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))]

    expected_backward = {
        'arg': [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]
    }

    const_min_value = constant(min_value, device=dev)
    const_max_value = constant(max_value, device=dev)

    _test_unary_op(precision, device_id, clip, x,
                   expected_forward, expected_backward,
                   {'min_value': const_min_value, 'max_value': const_max_value})
TENSORS = [
    ([[0, -0.1]]),
    ([[-100, -10], [-1, -0.1], [-0.01, -0.001],
      [0.001, 0.01], [0.1, 1], [10, 100]]),
]


@pytest.mark.parametrize("operand", TENSORS)
def test_op_sigmoid(operand, device_id, precision):
    s = 1.0 / (1.0 + np.exp(-AA(operand, dtype=PRECISION_TO_TYPE[precision])))
    expected_forward = [AA([s])]

    expected_backward = {
        'arg': [[s * (1 - s)]],
    }

    from .. import sigmoid
    _test_unary_op(precision, device_id, sigmoid, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_exp(operand, device_id, precision):
    e = np.exp(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = [AA([e])]

    expected_backward = {
        'arg': expected_forward,
    }

    from .. import exp
    _test_unary_op(precision, device_id, exp, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_abs(operand, device_id, precision):
    t = np.abs(AA(operand, dtype=PRECISION_TO_TYPE[precision]))

    expected_forward = [AA([t])]

    # For 0 NumPy gives a gradient non, while CNTK gives 0
    backward = operand / np.abs(operand)
    backward[np.isnan(backward)] = 0
    expected_backward = {
        'arg': [[backward]]
    }

    from .. import abs
    _test_unary_op(precision, device_id, abs, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_tanh(operand, device_id, precision):
    t = np.tanh(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = [AA([t])]

    expected_backward = {
        'arg': [[1 - t**2]],
    }

    from .. import tanh
    _test_unary_op(precision, device_id, tanh, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("shape", [(3, 9), (10, 20, 30)])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2, 0.5, 0.8])
def test_op_dropout(shape, dropout_rate, device_id, precision):
    from cntk import dropout

    count = 10
    resulted_non_zeros = 0

    # As the dropout node is stochastic, we run it a couple times and aggregate
    # over the results to get more stable tests.
    for i in range(count):
        value = np.ones(shape=shape, dtype=PRECISION_TO_TYPE[precision])

        a = I(shape=value.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

        dropout_node = dropout(a, dropout_rate=dropout_rate)

        value.shape = (1, 1) + value.shape
        forward_input = {a: value}

        forward, backward = cntk_eval(dropout_node,
                                 forward_input,
                                 precision,
                                 cntk_device(device_id),
                                 backward_pass=True)

        resulted_non_zeros += np.count_nonzero(forward[dropout_node.output])

    resulted_non_zeros /= count
    num_elements = np.multiply.reduce(shape)
    expected_non_zeros = num_elements * (1 - dropout_rate)
    max_off = 0.2 * num_elements

    assert(abs(resulted_non_zeros - expected_non_zeros) <
           max_off)


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 100])
def test_op_dropout_bad_input(dropout_rate):
    from cntk import dropout

    a = I(shape=(1, 2), dtype='float', needs_gradient=True, name='a')

    with pytest.raises(ValueError):
        dropout_node = dropout(a, dropout_rate=dropout_rate)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_sqrt(operand, device_id, precision):
    t = np.sqrt(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    t[np.isnan(t)] = 0
    expected_forward = [AA([t])]

    backward = 1 / (2 * t)

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import sqrt
    _test_unary_op(precision, device_id, sqrt, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_square(operand, device_id, precision):
    s = AA(operand, dtype=PRECISION_TO_TYPE[
           precision]) * AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_forward = [AA([s])]

    backward = 2 * AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import square

    _test_unary_op(precision, device_id, square, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_log(operand, device_id, precision):
    t = np.log(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    t[np.isnan(t)] = LOG_OF_EPS_IN_LOG
    t[np.isneginf(t)] = LOG_OF_EPS_IN_LOG

    expected_forward = [AA([t])]

    backward = 1 / AA(operand, dtype=PRECISION_TO_TYPE[precision])
    backward[np.isnan(backward)] = "inf"

    backward[np.isinf(backward)] = BACKWARD_RESULST_FOR_LOG_EPS
    backward[backward <= 0] = BACKWARD_RESULST_FOR_LOG_EPS

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import log

    _test_unary_op(precision, device_id, log, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_reciprocal(operand, device_id, precision):
    t = 1 / AA(operand, dtype=PRECISION_TO_TYPE[precision])
    t[np.isinf(t)] = 0
    expected_forward = [AA([t])]

    backward = -1 * t * t

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import reciprocal

    _test_unary_op(precision, device_id, reciprocal, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_relu(operand, device_id, precision):
    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_forward = [[np.maximum(np.zeros_like(t), t)]]

    expected_backward = {
        'arg': [[AA(t > np.zeros_like(t), dtype=int)]]
    }

    from cntk import relu

    _test_unary_op(precision, device_id, relu, operand,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_elu(operand, device_id, precision):
    elu_f  = np.vectorize(lambda x: np.exp(x) - 1.0 if x < 0 else x)
    elu_b  = np.vectorize(lambda x: np.exp(x) if x < 0 else 1.0)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [[elu_f(t)]]
    expected_backward = {
        'arg': [[elu_b(t)]]
    }

    from cntk import elu

    #BUGBUG: There is a bug in ElementSelect that cause nan in the output
    #        for float32.
    if PRECISION_TO_TYPE[precision] == np.float64:
        _test_unary_op(precision, device_id, elu, operand,
                       expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_leaky_relu(operand, device_id, precision):
    leaky_relu_f  = np.vectorize(lambda x: 0.01 * x if x < 0 else x)
    leaky_relu_b  = np.vectorize(lambda x: 0.01 if x < 0 else 1.0)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [[leaky_relu_f(t)]]
    expected_backward = {
        'arg': [[leaky_relu_b(t)]]
    }

    from cntk import leaky_relu

    _test_unary_op(precision, device_id, leaky_relu, operand,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_param_relu(operand, device_id, precision):
    dev = cntk_device(device_id)
    param_relu_f  = np.vectorize(lambda x: 0.5 * x if x < 0 else x)
    param_relu_b  = np.vectorize(lambda x: 0.5 if x < 0 else 1.0)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])
    a = AA(np.ones_like(t)*0.5, dtype=PRECISION_TO_TYPE[precision])
    alpha = constant(a, device=dev)

    expected_forward = [[param_relu_f(t)]]
    expected_backward = {
        'arg': [[param_relu_b(t)]]
    }

    from cntk import param_relu

    def prelu(x):
        return param_relu(alpha, x)

    _test_unary_op(precision, device_id, prelu, operand,
                    expected_forward, expected_backward)


SAMPLES = [  # 2 samples having 4 classes
    [1, 1, 2, 3],
    [0, 0, 0, 0],
    [3, 3, 4, 4]
]


@pytest.mark.parametrize("sample", SAMPLES)
def test_op_softmax(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    assert len(t.shape) == 1

    x_max = t - t.max()
    exp_x = np.exp(x_max)
    forward = exp_x / np.sum(exp_x)

    expected_forward = [AA([forward])]

    sample_length = len(forward)
    grad = np.zeros((sample_length, sample_length),
                    dtype=PRECISION_TO_TYPE[precision])

    for i in range(sample_length):
        for j in range(sample_length):
            if i == j:
                grad[i, j] = forward[i] * (1 - forward[i])
            else:
                grad[i, j] = -1 * forward[j] * forward[i]

    backward = grad.sum(axis=0)

    expected_backward = {
        'arg': [[backward]]
    }

    from cntk import softmax

    _test_unary_op(precision, device_id, softmax, sample,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("sample", SAMPLES)
def test_op_hardmax(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    t_max = t.max()

    forward = np.zeros_like(t, dtype=PRECISION_TO_TYPE[precision])

    for i, x in enumerate(t):
        if x == t_max:
            forward[i] = 1
            break

    expected_forward = [AA([forward])]

    expected_backward = {
        'arg': [[np.zeros_like(forward)]]
    }

    from cntk import hardmax

    _test_unary_op(precision, device_id, hardmax, sample,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("use_cudnn", [True, False])
@pytest.mark.parametrize("sample", SAMPLES)
def test_op_batch_normalization(use_cudnn, sample, device_id, precision):
    dtype = PRECISION_TO_TYPE[precision]
    epsilon = 0.00001
    dev = cntk_device(device_id)

    t = AA(sample, dtype=dtype).reshape(-1,1,1)
    mean = 1
    var = 2
    init_scale = 3
    init_bias = 4

    forward = [(x - mean) / np.sqrt(var + epsilon) * init_scale + init_bias for x in t]

    expected_forward = AA(forward)

    scale        = Parameter(init=AA([init_scale], dtype=dtype), device=dev)
    bias         = Parameter(init=AA([init_bias], dtype=dtype), device=dev)
    run_mean     = constant(mean, shape=(1), device=dev)
    run_variance = constant(var, shape=(1), device=dev)
    run_count = constant(0, device=dev)

    from cntk import batch_normalization

    a = I(shape=(1), dtype=dtype, needs_gradient=False, name='a')

    with pytest.warns(Warning):
        op = batch_normalization(a, scale, bias, run_mean, run_variance, False,
            #no running_count here, 
            epsilon=epsilon, use_cudnn_engine=use_cudnn)

    op_node = batch_normalization(a, scale, bias, run_mean, run_variance, running_count=run_count, spatial=False,
        epsilon=epsilon, use_cudnn_engine=use_cudnn)

    forward_input = {a: t}

    unittest_helper(op_node, forward_input, expected_forward, expected_backward=None, device_id=device_id, precision=precision)
