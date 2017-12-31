# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for evaluation operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
import cntk as C
from cntk.ops.tests.ops_test_utils import _test_binary_op, AA, precision, PRECISION_TO_TYPE,\
        unittest_helper

TARGET_OUT_PAIRS = [
    # (target_vector, output_vector)
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0.5, 0.5]], [[1., 2., 3., 4.]]),
    ([[0., 0.4, 0.3, 0.3]], [[2., 1., 1., 4.]])
]

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_cross_entropy_with_soft_max(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    ox = o - o.max()  # subtract max to avoid overflow
    exp_x = np.exp(ox)
    s_max = exp_x / np.sum(exp_x) # softmax function

    expected_forward = np.asarray(-np.sum(t * np.log(s_max, dtype=dt), dtype=dt))
    expected_forward.shape = (1,1,1) + expected_forward.shape

    s = np.sum(t, dtype=dt)
    backward = np.subtract(s_max * s, t)
    backward.shape = (1,) + backward.shape

    expected_backward = {
        'left_arg':  backward,
        'right_arg': [-1*o]
    }

    from cntk.losses import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_WITH_AXIS = [
    # (target_vector, output_vector, axis)
    ([[0., 0., 0., 1]],
     [[1., 2., 3., 4.]], -1),
    ([[0., 0., 0.5, 0.5]],
     [[1., 2., 3., 4.]], 1),
    ([[0., 0.4, 0.3, 0.3]],
     [[2., 1., 1., 4.]], 1),
    ([[0., 0., 0., 1],
      [0., 0., 1., 0.]],
     [[1., 2., 3., 4.],
      [1., 2., 3., 5.]], 1),
    ([[0., 0., 0., 1],
      [0., 1., 0., 0.]],
     [[1., 2., 3., 4.],
      [1., 7., 3., 5.]], 1)
]

@pytest.mark.parametrize("target_vector, output_vector, axis", TARGET_OUT_PAIRS_WITH_AXIS)
def test_op_cross_entropy_with_soft_max_and_axis(output_vector, target_vector, axis, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    x = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = []
    expected_backward_left = []
    expected_backward_right = []

    for sample, target in zip(x, t):
        ox = sample - sample.max()  # subtract max to avoid overflow
        exp_x = np.exp(ox)
        s_max = exp_x / np.sum(exp_x) # softmax function
        forward = np.asarray(-np.sum(target * np.log(s_max, dtype=dt), dtype=dt))
        expected_forward.append(forward.tolist())

        s = np.sum(target, dtype=dt)
        backward = np.subtract(s_max * s, target)

        expected_backward_left.append(backward.tolist())
        expected_backward_right.append(-1*sample)

    expected_forward = [np.reshape(AA(expected_forward, dtype=dt), (x.shape[0], 1))]
    expected_backward_left = AA(expected_backward_left, dtype=dt)

    expected_backward = {
        'left_arg':  [expected_backward_left],
        'right_arg': [expected_backward_right]
    }

    from cntk.losses import cross_entropy_with_softmax
    _test_binary_op(precision, device_id, cross_entropy_with_softmax,
                    output_vector, target_vector,
                    expected_forward, expected_backward, op_param_dict={'axis': axis})

@pytest.mark.parametrize("target_vector, output_vector", TARGET_OUT_PAIRS)
def test_op_squared_error(output_vector, target_vector, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    o = AA(output_vector, dtype=dt)
    t = AA(target_vector, dtype=dt)

    expected_forward = AA([np.sum((t - o)**2)])

    backward = 2 * np.subtract(o, t)
    expected_backward = {
        'left_arg':  [backward],
        'right_arg': [-1*backward]
    }

    from cntk.losses import squared_error
    _test_binary_op(precision, device_id, squared_error,
                    output_vector, target_vector,
                    expected_forward, expected_backward)

TARGET_OUT_PAIRS_CLASSIFICATION = [
    # (target_vector, output_vector)
    ([[1., 0., 0., 0]], [[1., 2., 3., 4.]]),
    ([[0., 0., 0., 1]], [[1., 2., 3., 4.]]),
]

LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS = [
    # (grad, value, output, gain)
    ([[-0.2121461],  [ 0.2121461]],  58.038055419921875, [1, 2], [7, 1]),
    ([[-0.14861868], [ 0.14861868]], 40.65847396850586,  [3, 4], [3, 1])
]

@pytest.mark.parametrize("grad, value, output, gain", LAMBDA_RANK_GRADIENTS_VALUES_AND_INPUTS)
def test_lambda_rank(grad, value, output, gain, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    score = AA(output, dtype=dt).reshape(-1,1,1)
    gain  = AA(gain, dtype=dt).reshape(-1,1,1)
    group = np.ones_like(score).reshape(-1,1,1)

    expected_value = AA(value, dtype=dt)
    expected_grad  = AA(grad, dtype=dt)

    from cntk.losses import lambda_rank

    g = C.input_variable((1,))
    s = C.input_variable((1,), needs_gradient=True)
    n = C.input_variable((1,))
    f = lambda_rank(s, n, g)

    actual_grad, actual_value = f.grad({s:score, n:gain, g:group}, [s], [f.output])

    assert np.allclose(actual_value, expected_value)
    assert np.allclose(actual_grad,  expected_grad)


NCE_EXPECTED_VALUES = [
    # (classes, xdim, batch, expected_value)
    (100,     50,  2, [ 3.52544 ,  5.671973]),
    (1000,   100,  4, [ 1.949046,  2.219169,  2.426618,  3.094275]),
    (10000,  200,  6, [ 1.494069,  1.569222,  1.628346,  1.64969 ,  1.673538,  1.755621]),
]

@pytest.mark.parametrize("classes, xdim, batch, expected_value", NCE_EXPECTED_VALUES)
def test_nce_loss(classes, xdim, batch, expected_value, device_id, precision):
    dt = PRECISION_TO_TYPE[precision]

    from cntk.losses import nce_loss
    import scipy

    x = C.input_variable(xdim, needs_gradient=True)
    y = C.input_variable(classes, is_sparse=True)

    x0 = np.arange(batch * xdim, dtype=dt).reshape((batch, xdim))/(batch * xdim)
    data = np.ones(batch, dtype=dt)
    indices = list(range(10,10*batch+1,10))
    indptr = list(range(batch+1))
    y0 = scipy.sparse.csr_matrix((data, indices, indptr), shape=(batch, classes))

    q = np.arange(classes, dtype=dt) + 1

    b = C.parameter((classes, 1), init=-np.log(classes))
    W = C.parameter((classes, C.InferredDimension), init=C.glorot_uniform(seed=98052))

    loss = C.nce_loss(W, b, x, y, q, seed=98052)
    v = loss.grad({x:x0, y:y0}, wrt=loss.parameters, as_numpy=False)
    for key in v:
        assert v[key].is_sparse, "gradient of nce_loss with respect to %s is not sparse"%key
    losses = np.zeros((100,batch))
    for i in range(100):
        losses[i,:] = loss.eval({x:x0, y:y0})
    assert np.allclose(np.mean(losses, axis=0), AA(expected_value))


@pytest.mark.parametrize("classes, xdim, batch, expected_value", NCE_EXPECTED_VALUES)
def test_nce_backward_indices(classes, xdim, batch, expected_value, device_id, precision):
    """
    Simple test that makes sure that the derivatives have the correct sparsity pattern
    """

    # ignore precision, only sparsity pattern matters for this test
    dt = np.float32

    from cntk.losses import nce_loss
    import scipy
    trials = 10

    # Establish baseline
    expected_count = np.zeros(classes)
    I = C.constant(np.eye(classes, dtype=dt))
    q = np.arange(classes, dtype=dt) + 1
    z = C.reduce_sum(C.times(C.random_sample(q, 32, True, seed=98052), I), axis=0)
    for i in range(trials):
        expected_count[np.nonzero(z.eval().ravel())] += 1

    # Set things up to measure the same thing with nce_loss

    x = C.input_variable(xdim, needs_gradient=True)
    y = C.input_variable(classes, is_sparse=True)

    x0 = np.arange(batch * xdim, dtype=dt).reshape((batch, xdim))/(batch * xdim)
    data = np.ones(batch, dtype=dt)
    indices = list(range(10,10*batch+1,10))
    indptr = list(range(batch+1))
    y0 = scipy.sparse.csr_matrix((data, indices, indptr), shape=(batch, classes))

    b = C.parameter((classes, 1))
    W = C.parameter((classes, C.InferredDimension))

    gb = np.zeros(classes)
    vb = C.input_variable((classes, 1), dtype=dt)
    Ib = C.constant(np.eye(1, dtype=dt))
    zb = C.times(vb, Ib)

    loss = C.nce_loss(W, b, x, y, q, seed=98052)
    for i in range(trials):
        v = loss.grad({x: x0, y: y0}, wrt=loss.parameters, as_numpy=False)
        gb[np.nonzero(zb.eval({vb: v[b]}).ravel())] += 1
    for i in range(classes):
        assert gb[i] == expected_count[i] or (i in indices and gb[i] == trials)

def test_weighted_binary_cross_entropy_with_reduced_sequences():
    pytest.skip("Skip this test until the cudaFree crash on exit is fixed for Windows")

    a = C.sequence.input_variable((), sequence_axis=C.Axis("a"))
    b = C.sequence.input_variable((), sequence_axis=C.Axis("b"))
    w = C.sequence.input_variable((), sequence_axis=C.Axis("w"))
    cd = C.weighted_binary_cross_entropy(C.sequence.first(a), C.sequence.first(b), C.sequence.first(w))
    data = np.random.random((4,)).astype(np.float32)
    cd.eval({a:data, b:data, w:data})