# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
import cntk as C
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, precision, PRECISION_TO_TYPE, cntk_device
from cntk.internal import sanitize_dtype_cntk
from cntk.internal.utils import _ones_like, eval

TENSOR_PAIRS = [
    ([30.], [10.]),
    ([[10.]], [[30.]]),
    ([[1.5, 2.1]], [[10., 20.]]),
    ([[100., 200.], [300., 400.], [10., 20.]],
     [[10., 20.], [30., 40.], [1., 2.]]),

    # Adding two 3x2 inputs of sequence length 1
    ([[30., 40.], [1., 2.], [0.1, 0.2]], [[10, 20], [3, 4], [-0.5, -0.4]]),
]

# -- plus operation tests --
TENSOR_PAIRS_SCALAR = TENSOR_PAIRS + [(left, np.random.rand()) for left, right
                                      in TENSOR_PAIRS]


@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS_SCALAR)
def test_op_plus(left_operand, right_operand, device_id, precision):
    expected_forward = AA([left_operand]) + AA([right_operand])

    if np.isscalar(right_operand):
        expected_backward = {
            'left_arg':  [[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]],
            # gradients are accumulated
            'right_arg': [AA([left_operand]).size]
        }
    else:
        expected_backward = {
            'left_arg':  [[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]],
            'right_arg': [[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in right_operand]]
        }
    from .. import plus
    _test_binary_op(precision, device_id, plus,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '+',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

def test_op_plus_sequences(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]
    operand = [AA([[1., 2.], [3., 4.]], dtype=dt_precision), AA([[5., 6.]], dtype=dt_precision)]
    root_gradient = [AA([[1., 1.], [1., 1.]], dtype=dt_precision), AA([[1., 1.]], dtype=dt_precision)]

    expected_forward = [AA([[2., 4.], [6., 8.]], dtype=dt_precision), AA([[10., 12.]], dtype=dt_precision)]
    expected_backward = [AA([[2., 2.], [2., 2.]], dtype=dt_precision), AA([[2., 2.]], dtype=dt_precision)]

    from .. import plus, sequence
    x = sequence.input_variable(shape=(2,), needs_gradient=True)
    z = x + x
    state, actual_forward = z.forward({x : operand}, [z.output], {z.output}, cntk_device(device_id))
    actual_backward = z.backward(state, {z.output : root_gradient}, [x])

    assert np.allclose(list(actual_forward.values())[0][0], expected_forward[0])
    assert np.allclose(list(actual_forward.values())[0][1], expected_forward[1])

    assert np.allclose(list(actual_backward.values())[0][0], expected_backward[0])
    assert np.allclose(list(actual_backward.values())[0][1], expected_backward[1])

def test_op_plus_gradient_accumulation(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    value = AA([[1]], dtype=dt_precision)

    from cntk import times_transpose, Axis
    a = C.input_variable(shape=(1,), dtype=dt_precision, needs_gradient=True, name='a')

    input_op = a + a

    expected_forward = AA([[2]], dtype=dt_precision)
    expected_backward = { a : [[2]], a : [[2]] }

    forward_input = {a: value}

    unittest_helper(input_op,
                    forward_input, expected_forward, expected_backward,
                    device_id=device_id, precision=precision)


SEQ_TENSOR_PAIRS = [
    # two inputs each having sequences of length 1 and 2
    ([[[30.]], [[40], [50]]],  # first batch with two sequences
     [[[3.]], [[4], [5]]]),  # second batch with two sequences

    #([[[30.,   0]], [[40,   1], [50,   2]]],  # first batch with two sequences
     #[[[3., -10]], [[4, -20], [5, -30]]]),  # second batch with two sequences
]


@pytest.mark.parametrize("left_batch, right_batch", SEQ_TENSOR_PAIRS)
def test_op_plus_var_sequences_input_input(left_batch, right_batch, device_id, precision):
    from .. import plus, sequence

    assert len(left_batch) == len(right_batch)
    expected_forward = [AA(left_batch[i]) + AA(right_batch[i])
                        for i in range(len(left_batch))]

    expected_backward = {
        'left': _ones_like(left_batch, PRECISION_TO_TYPE[precision]),
        'right': _ones_like(right_batch, PRECISION_TO_TYPE[precision])
    }

    left_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision])
                  for sample in left_batch]
    left_shape = left_value[0][0].shape
    right_value = [AA(sample, dtype=PRECISION_TO_TYPE[precision])
                   for sample in right_batch]
    right_shape = right_value[0][0].shape

    a = sequence.input_variable(shape=left_shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       needs_gradient=True,
                       name='a')

    b = sequence.input_variable(shape=right_shape,
                       dtype=sanitize_dtype_cntk(PRECISION_TO_TYPE[precision]),
                       needs_gradient=True,
                       name='b')

    input_op_input = plus(a, b)
    forward_input = {a: left_value, b: right_value}
    backward_input = {a: None, b: None}
    expected_backward = {
            a: expected_backward['left'], 
            b: expected_backward['right'], }
    unittest_helper(input_op_input,
                    forward_input, expected_forward,
                    expected_backward,
                    device_id, precision)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_minus(left_operand, right_operand, device_id, precision):
    expected_forward = AA([left_operand], dtype=PRECISION_TO_TYPE[
                           precision]) - AA([right_operand], dtype=PRECISION_TO_TYPE[precision])

    expected_backward = {
        'left_arg':  [[np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in left_operand]],
        'right_arg': [[-1 * np.ones_like(x, dtype=PRECISION_TO_TYPE[precision]) for x in right_operand]]
    }
    from .. import minus
    _test_binary_op(precision, device_id, minus,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '-',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_times(left_operand, right_operand, device_id, precision):
    expected_forward = AA([left_operand]) * AA([right_operand])

    expected_backward = {
        'left_arg':  [right_operand],
        'right_arg': [left_operand]
    }

    from .. import element_times
    _test_binary_op(precision, device_id, element_times,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '*',
                    left_operand, right_operand,
                    expected_forward, expected_backward)


@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_divide(left_operand, right_operand, device_id, precision):
    expected_forward = AA([left_operand]) / AA([right_operand])

    expected_backward = {
        'left_arg':  [[np.ones_like(x) / x for x in right_operand]],
        'right_arg': [-AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) / AA(right_operand, dtype=PRECISION_TO_TYPE[precision])**2]
    }

    from .. import element_divide
    _test_binary_op(precision, device_id, element_divide,
                    left_operand, right_operand,
                    expected_forward, expected_backward)

    _test_binary_op(precision, device_id, '/',
                    left_operand, right_operand,
                    expected_forward, expected_backward)

NEGATE_TENSORS = [
    ([30.]),
    ([[30.]]),
    ([[1.5, 2.1]]),
    ([[100., 200.], [300., 400.], [10., 20.]]),
    ([[30, 40], [1, 2], [0.1, 0.2]])
]


@pytest.mark.parametrize("operand", NEGATE_TENSORS)
def test_op_negate(operand, device_id, precision):
    t = -1 * AA(operand, dtype=PRECISION_TO_TYPE[precision])

    expected_forward = AA([t])

    expected_backward = {
        'arg': [-1 * np.ones_like(operand, PRECISION_TO_TYPE[precision])]
    }

    from cntk import negate

    _test_unary_op(precision, device_id, negate, operand,
                   expected_forward, expected_backward)

    _test_unary_op(precision, device_id, '-', operand,
                   expected_forward, expected_backward)


BATCH_TIMES_PAIRS = [(np.reshape(np.arange(8), (2, 2, 2)), np.reshape(np.arange(8), (2, 2, 2)))]
@pytest.mark.parametrize("left_operand, right_operand", BATCH_TIMES_PAIRS)
def test_op_batch_times(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    aa = AA(left_operand, dtype=dt_precision)
    bb = AA(right_operand, dtype=dt_precision)
    
    k, m, n = aa.shape[0], aa.shape[1], bb.shape[2]
    expected_forward = np.zeros((k, m, n))
    for x in range(k):
        expected_forward[x] = np.matmul(aa[x], bb[x])

    left_backward = np.zeros_like(aa)
    for x in range(k):
        left_backward[x, ...] = bb[x].sum(axis=-1)

    right_backward = np.zeros_like(bb)
    for x in range(k):
        transpose_axes = list(np.roll(np.arange(len(bb.shape[1:])), -1))
        sum_axes = tuple(np.arange(0, len(aa.shape) - len(bb.shape) + 1))
        right_backward[x, ...] = np.transpose(
            AA([aa[x].sum(axis=sum_axes)]), axes=transpose_axes)

    expected_backward = {
        'left_arg':  left_backward,
        'right_arg': right_backward
    }

    from cntk import times

    _test_binary_op(precision, device_id, times,
                    left_operand, right_operand, expected_forward, expected_backward, batch_size_greater_than_one=True)


@pytest.mark.parametrize("left_operand, right_operand", BATCH_TIMES_PAIRS)
def test_op_batch_times_with_inferred_axis(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]
    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)
        
    input1 = C.input_variable((2,2))
    input2 = C.input_variable((C.InferredDimension,2))
    z = C.times(input1, input2)
    actual_forward = z.eval({input1: a, input2: b}, device=cntk_device(device_id))

    expected_forward = np.ones((a.shape[0], a.shape[1], b.shape[2]))
    k = a.shape[0]
    for x in range(k):
        expected_forward[x, ...] = a[x].dot(b[x])

    assert np.allclose(actual_forward, expected_forward)


@pytest.mark.parametrize("left_operand, right_operand", BATCH_TIMES_PAIRS)
def test_op_batch_times_grad_with_beta_equals_to_one(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]
    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)
    
    root_gradient = np.ones_like(a)
    
    input1 = C.input_variable((2,2), needs_gradient=True)
    input2 = C.input_variable((2,2), needs_gradient=True)
    z = input1 + input2 + C.times(input1, input2)
    state, actual_forward = z.forward({input1: a, input2: b}, [z.output], {z.output}, cntk_device(device_id))
    actual_backwards = z.backward(state, {z.output: root_gradient}, [input1, input2])
    
    k = a.shape[0]
    left_backward = np.ones_like(a)
    for x in range(k):
        left_backward[x, ...] += b[x].sum(axis=-1)
    right_backward = np.ones_like(b)
    for x in range(k):
        transpose_axes = list(np.roll(np.arange(len(b.shape[1:])), -1))
        sum_axes = tuple(np.arange(0, len(a.shape) - len(b.shape) + 1))
        right_backward[x, ...] += np.transpose(
            AA([a[x].sum(axis=sum_axes)]), axes=transpose_axes)

    assert np.allclose(actual_backwards[input1], left_backward)
    assert np.allclose(actual_backwards[input2], right_backward)



# transpose_times currently only supports right operands of rank 1 or 2
TRANSPOSE_TIMES_PAIRS = [
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10.], [20.]]),
    ([[100., 200.]], [[-10.], [20.]]),
    ([[100., 200.], [300., 400.]], [[10.], [20.]]),
    ([[100., 200.], [-300., 400.]], [[10., 20.], [20., 30.]]),
    (np.reshape(np.arange(24), (4, 3, 2)),
     np.array([[1, 3], [2, 4]])),
]

# adding a rank 3 operand for times operation
TIMES_PAIRS = TRANSPOSE_TIMES_PAIRS + \
    list((np.reshape(np.arange(8), (2, 2, 2)), np.reshape(np.arange(8), (2, 2, 2))))


@pytest.mark.parametrize("left_operand, right_operand", TIMES_PAIRS)
def test_op_times(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)

    expected_forward = [np.tensordot(a, b, axes=len(b.shape) - 1)]

    left_backward = np.zeros_like(a)
    left_backward[...] = b.sum(axis=-1)

    right_backward = np.zeros_like(b)
    transpose_axes = list(np.roll(np.arange(len(b.shape)), -1))
    sum_axes = tuple(np.arange(0, len(a.shape) - len(b.shape) + 1))
    right_backward[...] = np.transpose(
        AA([a.sum(axis=sum_axes)]), axes=transpose_axes)

    expected_backward = {
        'left_arg':  [left_backward],
        'right_arg': [right_backward]
    }

    from cntk import times

    _test_binary_op(precision, device_id, times,
                    left_operand, right_operand, expected_forward, expected_backward)


@pytest.mark.parametrize("left_operand, right_operand", TRANSPOSE_TIMES_PAIRS)
def test_op_transpose_times(left_operand, right_operand, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    # tranpose right_operand to make product possible
    right_operand = np.transpose(right_operand).tolist()

    a = AA(left_operand, dtype=dt_precision)
    b = AA(right_operand, dtype=dt_precision)

    expected_forward = [np.dot(a, np.transpose(b))]

    left_backward = np.zeros_like(a)
    left_backward[...] = b.sum(axis=tuple(range(len(b.shape) - 1)))

    right_backward = np.zeros_like(b)
    right_backward[...] = a.sum(axis=tuple(range(len(a.shape) - 1)))

    expected_backward = {
        'left_arg':  [left_backward],
        'right_arg': [right_backward]
    }

    from cntk import times_transpose

    _test_binary_op(precision, device_id, times_transpose,
                    left_operand, right_operand, expected_forward, expected_backward)

def test_times_transpose_sequence_param(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    from cntk import times_transpose, parameter, sequence, Value
    dim = 5
    num_sequences = 2
    seq = [i for i in range(dim)]
    identity = np.identity(dim, dtype=dt_precision)
    input_data = Value.one_hot([seq]*num_sequences, dim, dtype=dt_precision)
    input_var  = sequence.input_variable(shape=(dim), needs_gradient=True, dtype=dt_precision)
    e = parameter(shape = (dim,), init = 1, dtype=dt_precision)
    z = times_transpose(e, input_var)
    e_grad = z.grad({input_var : input_data}, [e, input_var])

def test_op_times_sparse_grad(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    from cntk import times, times_transpose, parameter, reshape, Value, sequence
    dim = 5
    num_sequences = 2
    seq = [i for i in range(dim)]
    identity = np.identity(dim, dtype=dt_precision)
    input_data = Value.one_hot([seq]*num_sequences, dim, dtype=dt_precision)
    input_var  = sequence.input_variable(shape=(dim), is_sparse=True, needs_gradient=False, dtype=dt_precision)
    e = parameter(shape = (dim, dim), init = identity, dtype=dt_precision)
    z = reshape(times_transpose(e, times(input_var, e)), dim)
    e_grad = z.grad({input_var : input_data}, [e])

    assert np.allclose(e_grad, np.ones((dim,dim))*4)

def test_op_times_reduce_sequence_axis(device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]

    from cntk import times, Value, TIMES_REDUCE_SEQUENCE_AXIS_WITHOUT_INFERRED_INPUT_RANK
    from cntk import sequence
    dim = 10
    seq = [[0,1,2], [3], [4,5,6,7,8,9]]
    right_data = Value.one_hot(seq, dim, dtype=dt_precision)
    right_var = sequence.input_variable(shape=(dim), is_sparse=True, dtype=dt_precision)
    left_data = [AA([1,1,1],dtype=dt_precision), AA([1],dtype=dt_precision), AA([1,1,1,1,1,1],dtype=dt_precision)]
    left_var = sequence.input_variable(shape=(1), dtype=dt_precision)

    func = times(left_var, right_var, infer_input_rank_to_map=TIMES_REDUCE_SEQUENCE_AXIS_WITHOUT_INFERRED_INPUT_RANK)
    func2 = sequence.reduce_sum(times(left_var, right_var))

    assert func.dynamic_axes == func2.dynamic_axes

    _, forward_output = func.forward({left_var:left_data, right_var:right_data})
    
    actual_forward = forward_output[func.output]

    expected_forward = AA([[[1,1,1,0,0,0,0,0,0,0]],
                           [[0,0,0,1,0,0,0,0,0,0]],
                           [[0,0,0,0,1,1,1,1,1,1]]])
    
    assert np.allclose(actual_forward, expected_forward)

def test_per_dim_mean_var_norm():
    mean = np.asarray([2.], dtype=np.float32)
    inv_stddev = np.asarray([0.5], dtype=np.float32)
    x = C.input_variable((1,))
    func = C.per_dim_mean_variance_normalize(x, mean, inv_stddev)
    result = func.eval({x : np.asarray([[3.], [1.]], dtype=np.float32)})
    assert np.array_equal(result, [[.5], [-.5]])

def test_times_const_broadcast():
    x = C.input_variable((3,))
    a = C.constant(np.ones((3,), dtype=np.float32))
    y = C.times_transpose(a, x)
    result = y.eval({x:np.asarray([[1,2,3],[1,2,3]], dtype=np.float32)})
    assert np.array_equal(result, [[6], [6]])

def test_sequence_auto_broadcast():
    x = C.sequence.input((3,))
    y = C.input((3,))
    f = x * y
    result = f.eval({x:np.asarray([[1, 2, 3],[4, 5, 6]], dtype=np.float32),
                     y:np.asarray([[1, 2, 3]], dtype=np.float32)})
    assert np.array_equal(result[0], np.asarray([[1., 4., 9.],[4., 10., 18.]], dtype=np.float32))

def test_auto_broadcast_reconcile_issue():
    x = C.sequence.input((3,), name='x')
    y = C.input((3,), name='y')
    y2 = C.reconcile_dynamic_axes(y, x)
    inputs = y2.owner.inputs
    # check does the reconcile_dynamic_axes call trigger the auto broadcast
    assert len(inputs) == 2
    assert inputs[0].name == 'y' and inputs[1].name == 'x'

MEAN_VARIANCE_NORMALIZATION_DATA = [
    (np.array([[[0., 2.],     # Input tensor
                [4., 6.]],
               [[0., 4],
                [8., 12.]]]),
     False,                   # use_stats_across_channels
     False,                   # do_variance_scaling
     0.0,                     # epsilon
     np.array([[[-3., -1.],   # Output tensor
                [1., 3.]],
               [[-6., -2],
                [2., 6.]]])
     ),
    (np.array([[[0., 2.],     # Input tensor
                [4., 6.]],
               [[0., 4],
                [8., 12.]]]),
     False,                   # use_stats_across_channels
     True,                    # do_variance_scaling
     0.00001,                 # epsilon
     np.array([[[-1.34163487, -0.44721162],
                [ 0.44721162,  1.34163487]],
               [[-1.34163785, -0.44721264],
                [ 0.44721264,  1.34163785]]])
     ),
    (np.array([[[0., 2.],     # Input tensor
                [4., 6.]],
               [[0., 4],
                [8., 12.]]]),
     False,                   # use_stats_across_channels
     True,                    # do_variance_scaling
     0.01,                 # epsilon
     np.array([[[-1.33566761, -0.44522253],
                [ 0.44522253,  1.33566761]],
               [[-1.33864748, -0.44621584],
                [ 0.44621584,  1.33864748]]])
     ),
    (np.array([[[2., 2.],     # Input tensor
                [2., 2.]],
               [[2., 2],
                [2., 2.]]]),
     False,                   # use_stats_across_channels
     True,                   # do_variance_scaling
     0.00001,                    # epsilon
     np.array([[[0.0,  0.0],
                [0.0,  0.0]],
               [[0.0,  0.0],
                [0.0,  0.0]]])
     ),
]

@pytest.mark.parametrize("input_operand, use_stats_across_channels, do_variance_scaling, epsilon, output_ref", MEAN_VARIANCE_NORMALIZATION_DATA)
def test_op_mean_variance_normalization(input_operand, use_stats_across_channels, do_variance_scaling, epsilon, output_ref, device_id, precision):
    dt_precision = PRECISION_TO_TYPE[precision]
    input_ref = AA(input_operand, dtype=dt_precision)
    a = C.input_variable(shape=input_ref.shape,
                dtype=sanitize_dtype_cntk(precision),
                needs_gradient=False,
                name='a')
    norm_op = C.mean_variance_normalization(a, epsilon=epsilon, use_stats_across_channels=use_stats_across_channels, do_variance_scaling=do_variance_scaling)
    output_test = norm_op.eval({a:input_ref}, device=cntk_device(device_id))

    assert np.allclose(output_test, output_ref, atol=1e-4)