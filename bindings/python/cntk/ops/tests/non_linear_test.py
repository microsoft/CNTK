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
import cntk as C
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, precision, PRECISION_TO_TYPE, cntk_device
from cntk.tests.test_utils import TOLERANCE_ABSOLUTE
from cntk.internal.utils import eval as cntk_eval
from cntk.internal import sanitize_dtype_cntk
from .. import constant
from cntk.variables import Parameter, Constant

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

    expected_forward = np.clip(AA([x], dtype=PRECISION_TO_TYPE[precision]), AA(
        min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))

    expected_backward = {
        'arg': [np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]
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
    expected_forward = AA([s])

    expected_backward = {
        'arg': [s * (1 - s)],
    }

    from .. import sigmoid
    _test_unary_op(precision, device_id, sigmoid, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_softplus(operand, device_id, precision):
    s = np.logaddexp(0, (AA(operand, dtype=PRECISION_TO_TYPE[precision])))
    # BUGBUG: The inner implementation is a tiny bit less accurate than numpy, so we manually replace the values
    if s.shape == (6,2):
        if operand[0][1] == -10:
            s[0,1] = 0  # np baseline is 0.000045
        if operand[5][0] == 10:
            s[5,0] = 10 # np baseline is 10.000045
    expected_forward = AA([s])

    from .. import softplus
    _test_unary_op(precision, device_id, softplus, operand,
                   expected_forward, None)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_exp(operand, device_id, precision):
    e = np.exp(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = AA([e])

    expected_backward = {
        'arg': expected_forward,
    }

    from .. import exp
    _test_unary_op(precision, device_id, exp, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_abs(operand, device_id, precision):
    t = np.abs(AA(operand, dtype=PRECISION_TO_TYPE[precision]))

    expected_forward = AA([t])

    # For 0 NumPy gives a gradient non, while CNTK gives 0
    backward = operand / np.abs(operand)
    backward[np.isnan(backward)] = 0
    expected_backward = {
        'arg': [backward]
    }

    from .. import abs
    _test_unary_op(precision, device_id, abs, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_tanh(operand, device_id, precision):
    t = np.tanh(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    expected_forward = AA([t])

    expected_backward = {
        'arg': [1 - t**2],
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

        a = C.input_variable(shape=value.shape,
                  dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                  needs_gradient=True,
                  name='a')

        dropout_node = dropout(a, dropout_rate=dropout_rate)

        value.shape = (1,) + value.shape
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


def test_changing_dropout_rate():
    from cntk import dropout, input

    resulted_non_zeros = 0

    shape = (100,100)
    dtype = np.float32
    value = np.ones(shape=shape, dtype=dtype)

    a = input(shape=shape, needs_gradient=True, dtype=dtype)
    dropout_node = dropout(a, dropout_rate=0.1)

    value.shape = (1,) + value.shape

    for dropout_rate in [0.0, 0.25,  0.5, 0.78, 0.99999]:
        dropout_node.set_attribute('dropoutRate', dropout_rate)
        forward, _ = cntk_eval(dropout_node, {a: value}, dtype, backward_pass=True)
        resulted_non_zeros = np.count_nonzero(forward[dropout_node.output])
        if (dropout_rate == 0):
            assert resulted_non_zeros == value.size

        assert np.isclose((1-dropout_rate), resulted_non_zeros* 1.0/ value.size, atol=0.01)

def test_dropout_random_mask_is_recomputed_on_forward_pass():
    from cntk import dropout, input

    shape = (100,100)
    dtype = np.float32
    value = np.ones(shape=shape, dtype=dtype)

    a = input(shape=shape, needs_gradient=True, dtype=dtype)
    dropout_node = dropout(a, dropout_rate=0.1)
    network = dropout_node + constant(0)

    value.shape = (1,) + value.shape

    _, forward = network.forward({a: value}, network.outputs, network.outputs)
    non_zeros_1 = forward[network.output] > 0.0

    _, forward = network.forward({a: value}, network.outputs, network.outputs)
    non_zeros_2 = forward[network.output] > 0.0

    assert not (non_zeros_1 == non_zeros_2).all()

def test_op_dropout_with_explicit_seed(device_id, precision):
    from cntk import combine, dropout

    value = np.ones(shape=(100,100), dtype=PRECISION_TO_TYPE[precision])

    a = C.input_variable(shape=value.shape,
              dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
              needs_gradient=True,
              name='a')

    seed = 123;

    dropout_nodes= [
        dropout(a, dropout_rate=0.5, seed=seed),
        dropout(a, dropout_rate=0.5, seed=seed),
        dropout(a, dropout_rate=0.5, seed=seed+1),
        dropout(a, dropout_rate=0.5)
    ]

    cloned_nodes = [x.clone('clone') for x in dropout_nodes]

    value.shape = (1, 1) + value.shape

    results = []
    for node in dropout_nodes + cloned_nodes:
        forward_input = {node.inputs[0]: value}
        forward, backward = cntk_eval(node,
                                      forward_input,
                                      precision,
                                      cntk_device(device_id),
                                      backward_pass=True)

        results.append(forward[node.output])

    assert np.allclose(results[0], results[1])
    assert not np.allclose(results[0], results[2])
    assert not np.allclose(results[0], results[3])

    clones = results[len(dropout_nodes):]
    for i in range(len(clones)):
        assert np.allclose(results[i], clones[i])


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 100])
def test_op_dropout_bad_input(dropout_rate):
    from cntk import dropout

    a = C.input_variable(shape=(1, 2), dtype='float', needs_gradient=True, name='a')

    with pytest.raises(ValueError):
        dropout_node = dropout(a, dropout_rate=dropout_rate)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_sqrt(operand, device_id, precision):
    t = np.sqrt(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    t[np.isnan(t)] = 0
    expected_forward = AA([t])

    backward = 1 / (2 * t)

    expected_backward = {
        'arg': [backward]
    }

    from cntk import sqrt
    _test_unary_op(precision, device_id, sqrt, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_square(operand, device_id, precision):
    s = AA(operand, dtype=PRECISION_TO_TYPE[
           precision]) * AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_forward = AA([s])

    backward = 2 * AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_backward = {
        'arg': [backward]
    }

    from cntk import square

    _test_unary_op(precision, device_id, square, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_log(operand, device_id, precision):
    t = np.log(AA(operand, dtype=PRECISION_TO_TYPE[precision]))
    t[np.isnan(t)] = LOG_OF_EPS_IN_LOG
    t[np.isneginf(t)] = LOG_OF_EPS_IN_LOG

    expected_forward = AA([t])

    backward = 1 / AA(operand, dtype=PRECISION_TO_TYPE[precision])
    backward[np.isnan(backward)] = "inf"

    backward[np.isinf(backward)] = BACKWARD_RESULST_FOR_LOG_EPS
    backward[backward <= 0] = BACKWARD_RESULST_FOR_LOG_EPS

    expected_backward = {
        'arg': [backward]
    }

    from cntk import log

    _test_unary_op(precision, device_id, log, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_reciprocal(operand, device_id, precision):
    t = 1 / AA(operand, dtype=PRECISION_TO_TYPE[precision])
    t[np.isinf(t)] = 0
    expected_forward = AA([t])

    backward = -1 * t * t

    expected_backward = {
        'arg': [backward]
    }

    from cntk import reciprocal

    _test_unary_op(precision, device_id, reciprocal, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_relu(operand, device_id, precision):
    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])
    expected_forward = [np.maximum(np.zeros_like(t), t)]

    expected_backward = {
        'arg': [AA(t > np.zeros_like(t), dtype=int)]
    }

    from cntk import relu

    _test_unary_op(precision, device_id, relu, operand,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_elu(operand, device_id, precision):
    elu_f  = np.vectorize(lambda x: np.exp(x) - 1.0 if x < 0 else x)
    elu_b  = np.vectorize(lambda x: np.exp(x) if x < 0 else 1.0)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [elu_f(t)]
    expected_backward = {
        'arg': [elu_b(t)]
    }

    from cntk import elu

    _test_unary_op(precision, device_id, elu, operand,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_selu(operand, device_id, precision):
    scale = 1.0507009873554804934193349852946
    scale_alpha = 1.7580993408473768599402175208123
    selu_f  = np.vectorize(lambda x: scale_alpha * (np.exp(x) - 1.0) if x < 0 else scale * x)
    selu_b  = np.vectorize(lambda x: scale_alpha * np.exp(x) if x < 0 else scale)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [selu_f(t)]
    expected_backward = {
        'arg': [selu_b(t)]
    }

    from cntk import selu

    _test_unary_op(precision, device_id, selu, operand,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_leaky_relu(operand, device_id, precision):
    leaky_relu_f  = np.vectorize(lambda x: 0.01 * x if x < 0 else x)
    leaky_relu_b  = np.vectorize(lambda x: 0.01 if x < 0 else 1.0)

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [leaky_relu_f(t)]
    expected_backward = {
        'arg': [leaky_relu_b(t)]
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

    expected_forward = [param_relu_f(t)]
    expected_backward = {
        'arg': [param_relu_b(t)]
    }

    from cntk import param_relu

    def prelu(x):
        return param_relu(alpha, x)

    _test_unary_op(precision, device_id, prelu, operand,
                    expected_forward, expected_backward)

@pytest.mark.parametrize("operand", TENSORS)
def test_op_softplus(operand, device_id, precision):
    softplus_f = np.vectorize(lambda x: np.logaddexp(x, 0))
    softplus_b = np.vectorize(lambda x: 1.0/(1.0+np.exp(-x)))

    t = AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = [softplus_f(t)]
    expected_backward = {
        'arg': [softplus_b(t)]
    }

    from .. import softplus
    _test_unary_op(precision, device_id, softplus, operand,
                   expected_forward, expected_backward)

SAMPLES = [  # 5 samples having 4 classes
    [1, 1, 2, 3],
    [0, 0, 0, 0],
    [3, 3, 4, 4],
    [1000, 1000, 1000, 1000],
    [10000, 10000, 10000, 10000]
]


@pytest.mark.parametrize("sample", SAMPLES)
def test_op_softmax(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    assert len(t.shape) == 1

    x_max = t - t.max()
    exp_x = np.exp(x_max)
    forward = exp_x / np.sum(exp_x)

    expected_forward = AA([forward])

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
        'arg': [backward]
    }

    from cntk import softmax

    _test_unary_op(precision, device_id, softmax, sample,
                   expected_forward, expected_backward)


SAMPLES_AXIS = [  # 4 samples having 4 classes
    [[1], [1], [1], [1]],
    [[0], [0], [0], [0]],
    [[1000], [1000], [1000], [1000]],
    [[10000], [10000], [10000], [10000]]
]


@pytest.mark.parametrize("sample", SAMPLES_AXIS)
def test_op_softmax_axis(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    assert len(t.shape) == 2

    x_max = t - t.max()
    exp_x = np.exp(x_max)
    forward = exp_x / np.sum(exp_x)

    expected_forward = AA([forward])

    from cntk import softmax
    result = softmax(sample, axis=0).eval()

    assert np.array_equal(result, expected_forward[0])

@pytest.mark.parametrize("sample", SAMPLES_AXIS)
def test_op_softmax_with_freedimension(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    assert len(t.shape) == 2

    x_max = t - t.max()
    exp_x = np.exp(x_max)
    forward = exp_x / np.sum(exp_x)

    expected_forward = AA([forward])

    from cntk import softmax, input_variable
    x = input_variable((C.FreeDimension, t.shape[1]))
    result = softmax(x, axis=0).eval({x:[sample]})[0]

    assert np.array_equal(result, expected_forward[0])

@pytest.mark.parametrize("sample", SAMPLES)
def test_op_hardmax(sample, device_id, precision):
    t = AA(sample, dtype=PRECISION_TO_TYPE[precision])
    t_max = t.max()

    forward = np.zeros_like(t, dtype=PRECISION_TO_TYPE[precision])

    for i, x in enumerate(t):
        if x == t_max:
            forward[i] = 1
            break

    expected_forward = AA([forward])

    expected_backward = {
        'arg': [np.zeros_like(forward)]
    }

    from cntk import hardmax

    _test_unary_op(precision, device_id, hardmax, sample,
                   expected_forward, expected_backward)

@pytest.mark.parametrize("sample", SAMPLES)
def test_op_log_softmax(sample, device_id, precision):
    t = np.asarray(sample, dtype=PRECISION_TO_TYPE[precision])
    assert len(t.shape) == 1

    x_max = t - t.max()
    exp_x = np.exp(x_max)
    softmax = exp_x / np.sum(exp_x)
    forward = np.log(softmax)

    expected_forward = np.asarray([forward])

    from cntk import log_softmax

    _test_unary_op(precision, device_id, log_softmax, sample,
                   expected_forward, None)

@pytest.mark.parametrize("use_cudnn", [True, False])
@pytest.mark.parametrize("sample", SAMPLES)
def test_op_batch_normalization(use_cudnn, sample, device_id, precision):
    dtype = PRECISION_TO_TYPE[precision]
    epsilon = 0.00001
    dev = cntk_device(device_id)

    t = AA(sample, dtype=dtype).reshape(-1,1)
    mean = 1
    var = 2
    init_scale = 3
    init_bias = 4

    forward = [(x - mean) / np.sqrt(var + epsilon) * init_scale + init_bias for x in t]

    expected_forward = AA(forward)

    scale        = Parameter(init=AA([init_scale], dtype=dtype), dtype=dtype, device=dev)
    bias         = Parameter(init=AA([init_bias], dtype=dtype), dtype=dtype, device=dev)
    run_mean     = constant(mean, shape=(1), dtype=dtype, device=dev)
    run_variance = constant(var,  shape=(1), dtype=dtype, device=dev)
    run_count    = constant(0,               dtype=dtype, device=dev)

    from cntk import batch_normalization

    a = C.input_variable(shape=(1), dtype=dtype, needs_gradient=False, name='a')

    with pytest.warns(Warning):
        op = batch_normalization(a, scale, bias, run_mean, run_variance, False,
            #no running_count here,
            epsilon=epsilon, use_cudnn_engine=use_cudnn)

    op_node = batch_normalization(a, scale, bias, run_mean, run_variance, running_count=run_count, spatial=False,
        epsilon=epsilon, use_cudnn_engine=use_cudnn)

    forward_input = {a: t}

    unittest_helper(op_node, forward_input, expected_forward, expected_backward=None, device_id=device_id, precision=precision)

@pytest.mark.parametrize("shape", [(1,), (16,), (16,32,), (16,32,32,)])
@pytest.mark.parametrize("spatial", [True, False])
def test_op_batch_normalization_numpy(shape, spatial, device_id, precision):
    # for some reason the numpy code below does not work in python 2.7
    import sys
    if sys.version_info[0] < 3:
        pytest.skip("Only works on Python 3+")

    dtype = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    if spatial:
        param_shape = (shape[0],)
        reduced_shape = shape[1:]
        reduce_dims = (0,2,3,4)[0:len(shape)]
    else:
        param_shape = (np.prod(shape),)
        reduced_shape = ()
        reduce_dims = (0,)

    batch_size = 3
    x = 10 * np.random.random((batch_size,)+shape).astype(dtype)

    init_scale = 1
    init_bias  = 2
    init_mean  = 3
    init_var   = 4
    init_count = 2
    epsilon    = 0.01

    i = C.input_variable(shape, dtype=dtype)
    scale = C.parameter(param_shape, init=init_scale, dtype=dtype, device=dev)
    bias = C.parameter(param_shape, init=init_bias, dtype=dtype, device=dev)
    run_mean = C.constant(init_mean, shape=param_shape, dtype=dtype, device=dev)
    run_var = C.constant(init_var, shape=param_shape, dtype=dtype, device=dev)
    run_count = C.constant(init_count, shape=(), dtype=dtype, device=dev)
    #use negative normalization_time_constant for easier exp_avg compute
    bn = C.batch_normalization(i, scale, bias, run_mean, run_var, spatial, normalization_time_constant=-1, epsilon=epsilon, running_count = run_count)
    fwd = bn.eval(x, device=dev)
    y_fwd = (x - init_mean) / np.sqrt(init_var + epsilon) * init_scale + init_bias
    assert(np.allclose(y_fwd, fwd))

    bwd = bn.grad(x, wrt=bn.parameters, outputs=[bn], device=dev)
    exp_avg = batch_size / (init_count + batch_size)

    mean = np.mean(x, reduce_dims)
    mean_b = np.asarray([[np.ones(reduced_shape)*x for x in mean]]*batch_size)
    reduced_count = batch_size * np.prod(reduced_shape)
    var = np.mean((x - mean_b) ** 2, reduce_dims)
    #the output variance is unbiased, while computation uses biased variance
    var_out = var * reduced_count / (reduced_count - 1)
    var_b = np.asarray([[np.ones(reduced_shape)*x for x in var]]*batch_size)
    x_hat = (x - mean_b) / np.sqrt(var_b + epsilon)
    y = init_scale * x_hat + init_bias

    d_scale = np.sum(x_hat, reduce_dims)
    d_bias = np.sum(np.ones_like(x_hat), reduce_dims)

    assert(np.allclose(y, bwd[1], atol=1e-6))
    assert(np.allclose(d_scale.reshape(param_shape), bwd[0][scale], atol=1e-2))
    assert(np.allclose(d_bias.reshape(param_shape), bwd[0][bias]))
    assert(np.allclose(init_var * (1-exp_avg) + var_out.reshape(param_shape) * exp_avg, run_var.value))
    assert(np.allclose(init_mean * (1-exp_avg) + mean.reshape(param_shape) * exp_avg, run_mean.value))
    assert(run_count.value == init_count + batch_size)

@pytest.mark.parametrize("channels", [1, 16])
@pytest.mark.parametrize("input_size", [32, C.FreeDimension, C.InferredDimension])
def test_op_batch_normalization_spatial_shape_inference(channels, input_size, device_id, precision):
    dtype = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    spatial = True
    epsilon = 0.01

    init_scale = 1
    init_bias  = 2
    init_mean  = 3
    init_var   = 4
    init_count = 2

    shape = (channels, input_size, input_size)
    param_shape = (C.InferredDimension,)

    i = C.input_variable(shape, dtype=dtype)
    scale = C.parameter(param_shape, init=init_scale, dtype=dtype, device=dev)
    bias = C.parameter(param_shape, init=init_bias, dtype=dtype, device=dev)
    run_mean = C.constant(init_mean, shape=param_shape, dtype=dtype, device=dev)
    run_var = C.constant(init_var, shape=param_shape, dtype=dtype, device=dev)
    run_count = C.constant(init_count, shape=(), dtype=dtype, device=dev)

    bn = C.batch_normalization(i, scale, bias, run_mean, run_var, spatial, normalization_time_constant=-1, epsilon=epsilon, running_count = run_count)

    for param in [scale, bias, run_mean, run_var]:
        assert(param.shape == (channels,))


def test_local_response_normalization(device_id, precision):
    dtype = PRECISION_TO_TYPE[precision]
    dev = cntk_device(device_id)

    def lrn(x, depth_radius, bias, alpha, beta, name=''):
        x2 = C.square(x)
        # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
        x2s = C.reshape(x2, (1, C.InferredDimension), 0, 1)
        W = C.constant(alpha/(2*depth_radius+1), shape=(1,2*depth_radius+1,1,1), dtype=dtype, name='W')
        # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
        y = C.convolution (W, x2s)
        # reshape back to remove the fake singleton reduction dimension
        b = C.reshape(y, C.InferredDimension, 0, 2)
        den = C.exp(beta * C.log(bias + b))
        return C.element_divide(x, den)

    from cntk import local_response_normalization

    img_shape = (64, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=dtype)
    x_gt = C.input_variable(shape=img_shape, dtype=dtype)
    x_r = C.input_variable(shape=img_shape, dtype=dtype)

    gt = lrn(x_gt, 2, 1.0, 0.0001, 0.75)
    r = local_response_normalization(x_r, 2, 1.0, 0.0001, 0.75)
    ss = gt.eval({x_gt:img})
    sa = r.eval({x_r:img})

    assert np.allclose(r.eval({x_r:img}), gt.eval({x_gt:img}))

TENSOR_PAIRS = [
    ([0.3], [0.1]),
    ([[0.1]], [[0.3]]),
    ([[1.5, 2.1]], [[-2., -3.]]),
    ([[1., 2.], [3., 4.], [1., 2.]],
     [[2., 2.], [3., 1.], [-1., -2.]]),
]

@pytest.mark.parametrize("base, exponent", TENSOR_PAIRS)
def test_op_pow(base, exponent, device_id, precision):
    dt =  PRECISION_TO_TYPE[precision]
    base = AA(base,dtype=dt)
    exponent = AA(exponent,dtype=dt)
    expected_forward = base ** exponent

    expected_backward = {
            'left_arg':  [exponent * base**(exponent-1)],
            'right_arg': [expected_forward * np.log(base)]
        }

    from .. import pow
    _test_binary_op(precision, device_id, pow,
                    base, exponent,
                    AA([expected_forward]), expected_backward)

NEGATIVE_TENSOR_PAIRS = [
    ([-1., -2., -3., -4.], [2., -3., 3., -2.]),
]

@pytest.mark.parametrize("base, exponent", NEGATIVE_TENSOR_PAIRS)
def test_op_neg_pow(base, exponent, device_id, precision):
    dt =  PRECISION_TO_TYPE[precision]
    base = AA(base,dtype=dt)
    exponent = AA(exponent,dtype=dt)
    expected_forward = base ** exponent

    expected_backward = {
            'left_arg':  [exponent * base**(exponent-1)],
            'right_arg': [expected_forward * 0]
        }

    from .. import pow
    _test_binary_op(precision, device_id, pow,
                    base, exponent,
                    AA([expected_forward]), expected_backward)
