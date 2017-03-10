# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for reshaping operations.
"""

import numpy as np
import pytest
import cntk as C
from .ops_test_utils import cntk_device
from cntk.axis import Axis
from .. import constant

def test_times_2d_sparse_operand(device_id):
    from .. import times

    dev = cntk_device(device_id)

    vocab_size = 6
    sample_shape = (2, vocab_size)
    input_sparse_indices = [[1, 3], [2, 4], [0, 2]]
    input_data = C.Value.one_hot(input_sparse_indices, sample_shape, device=dev)

    a = C.input_variable(shape=sample_shape, is_sparse=True, needs_gradient=True, name='a')
    w = C.parameter(init=np.eye(vocab_size, dtype=np.float32), device=dev)
    a_dense = times(a, w)

    # TODO: Also test the results from grad
    grad = a_dense.grad({a : input_data}, [w, a], as_numpy=False, device=dev)

    res = a_dense.eval({a : input_data}, device=dev)
    assert np.array_equal(res, [[[[0, 1, 0, 0, 0, 0], [ 0, 0, 0, 1, 0, 0]]], [[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]], [[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]])

    a_no_sequence = C.input_variable(shape=sample_shape, is_sparse=True, name='a', dynamic_axes=[C.Axis.default_batch_axis()])
    c = C.constant(value=np.eye(vocab_size, dtype=np.float32), device=dev)
    a_no_sequence_dense = times(a_no_sequence, c)
    res = a_no_sequence_dense.eval({a_no_sequence : input_data}, device=dev)
    assert np.array_equal(res, [[[[0, 1, 0, 0, 0, 0], [ 0, 0, 0, 1, 0, 0]]], [[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]], [[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]])


def test_times_2d_sparse_sequence_operand(device_id):
    from .. import times

    dev = cntk_device(device_id)

    vocab_size = 6
    sample_shape = (2, vocab_size)
    input_sparse_indices = [[1, 3, 4, 2, 0, 5], [2, 4], [0, 2]]
    input_data = C.Value.one_hot(input_sparse_indices, sample_shape, device=dev)

    a = C.input_variable(shape=sample_shape, is_sparse=True, needs_gradient=True, name='a')
    w = C.parameter(init=np.eye(vocab_size, dtype=np.float32), device=dev)
    a_dense = times(a, w)

    # TODO: Also test the results from grad
    grad = a_dense.grad({a : input_data}, [w, a], as_numpy=False, device=dev)

    res = a_dense.eval({a : input_data}, device=dev)
    assert np.array_equal(res[0], [[[0, 1, 0, 0, 0, 0], [ 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]])
    assert np.array_equal(res[1], [[[0, 0, 1, 0, 0, 0], [ 0, 0, 0, 0, 1, 0]]])
    assert np.array_equal(res[2], [[[1, 0, 0, 0, 0, 0], [ 0, 0, 1, 0, 0, 0]]])


def test_training_2d_sparse_sequence_operand(device_id):
    from .. import times, cross_entropy_with_softmax

    dev = cntk_device(device_id)
    vocab_size = 6
    additional_axis_dim = 2
    out_dim = 4
    w_init = np.float32(np.random.rand(vocab_size, out_dim))
    input_shape = (additional_axis_dim, vocab_size)
    label_shape = (additional_axis_dim, out_dim)

    def create_trainer(use_sparse, device):
        a = C.input_variable(shape=input_shape, is_sparse=use_sparse, name='input')
        w = C.parameter(init=w_init, device=dev)
        z = times(a, w)
    
        l = C.input_variable(shape=label_shape, is_sparse=use_sparse, name='label')
        loss = cross_entropy_with_softmax(z, l, axis=-1)
        trainer = C.Trainer(z, (loss, None), C.sgd(z.parameters, lr=C.learning_rate_schedule(0.007, C.UnitType.sample)))
        return (a, l, w, trainer)

    # Run with sparse inputs
    input_sparse_indices = [[1, 3, 4, 2, 0, 5], [2, 4], [0, 2]]
    input_data = C.Value.one_hot(input_sparse_indices, input_shape, device=dev)
    label_sparse_indices = [[1, 3, 0, 2, 1, 0], [2, 1], [1, 3]]
    label_data = C.Value.one_hot(label_sparse_indices, label_shape, device=dev)

    input_var, label_var, weights, trainer = create_trainer(use_sparse=True, device=dev)
    trainer.train_minibatch({input_var:input_data, label_var:label_data}, device=dev)
    weights_with_sparse_input = weights.value

    # Run with dense inputs
    input_data = [np.float32(np.asarray([[[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]])), np.float32(np.asarray([[[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]])), np.float32(np.asarray([[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]))]
    label_data = [np.float32(np.asarray([[[0, 1, 0, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [1, 0, 0, 0]]])), np.float32(np.asarray([[[0, 0, 1, 0], [0, 1, 0, 0]]])), np.float32(np.asarray([[[0, 1, 0, 0], [0, 0, 0, 1]]]))]

    input_var, label_var, weights, trainer = create_trainer(use_sparse=False, device=dev)
    trainer.train_minibatch({input_var:input_data, label_var:label_data}, device=dev)
    weights_with_dense_input = weights.value

    assert np.allclose(weights_with_sparse_input, weights_with_dense_input)
