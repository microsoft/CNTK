# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the sequence_softmax.
"""

import numpy as np
import pytest
import cntk as C
from .. import *
from cntk.losses import *
from ...axis import Axis
from ... import sequence, input
from .ops_test_utils import cntk_device
from cntk.tests.test_utils import _to_dense, _to_csr


def test_lstm_over_lstm_thought_vectors(device_id):
    dev = cntk_device(device_id)
    input_vocab_size=3
    emb_dim = 2
    hidden_dim = 2
    num_labels = 2
    x_seq_input = C.sequence.input((C.FreeDimension, input_vocab_size), is_sparse=True, name='features')
    x_secondary_seq_lens = C.input((), name='sequence_lengths')
    label_seq_input = C.sequence.input(num_labels, is_sparse=True, sequence_axis=Axis('label_sequence'), name='labels')
    with C.default_options(initial_state=0.1):
        model = C.layers.Embedding(emb_dim, name='embed')(x_seq_input)
        model = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(model)
        model = C.sequence.last(model)
        model = C.to_sequence(model, x_secondary_seq_lens)
        model = C.reconcile_dynamic_axes(model, label_seq_input)
        model = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(model)
        model = C.layers.Dense(num_labels, name='classify')(model)

    z = model
    ce = C.cross_entropy_with_softmax(z, label_seq_input)

    seq1_data = [[[0, 1, 1], [0, 1, 0], [1, 0, 0]], [[1, 1, 0], [0, 0, 1], [1, 0, 1]], [[1, 0, 0], [0, 0, 1], [1, 1, 0]]]
    csr_seq1 = _to_csr(seq1_data)
    ndarrayview1 = C.NDArrayView.from_csr(csr_seq1, shape=(3, 3, 3), device=C.cpu())
    seq2_data = [[[0, 0, 1], [0, 1, 1], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 0, 0]]]
    csr_seq2 = _to_csr(seq2_data)
    ndarrayview2 = C.NDArrayView.from_csr(csr_seq2, shape=(2, 3, 3), device=C.cpu())
    x_seq_data = C.Value.create(C.sequence.input((3, 3), is_sparse=True), [ndarrayview1, ndarrayview2], device=C.cpu()).data
    x_seq_lens_data = np.asarray([3, 2], dtype=np.float32)

    seq1_label_data = [[0, 1], [0, 1], [1, 0]]
    seq2_label_data = [[1, 0], [0, 1]]
    label_seq_data = [_to_csr(seq1_label_data), _to_csr(seq2_label_data)]
    param_grads, loss_result = ce.grad({x_seq_input : x_seq_data, x_secondary_seq_lens : x_seq_lens_data, label_seq_input : label_seq_data},
                                       wrt=ce.parameters, outputs=[ce], as_numpy=False)
    
    loss_result = loss_result.as_sequences()
    assert np.allclose(loss_result[0], [[0.703254], [0.701883], [0.683452]], atol=0.03)
    assert np.allclose(loss_result[1], [[0.682687], [0.696831]], atol=0.03)


def test_sequence_max():
  np.random.seed(0)
  a = np.float32(np.random.rand(20,100,8))
  src = sequence.input(shape=(8), sequence_axis=Axis("Seq"))
  out = sequence.reduce_max(src)
  val = out.eval({src:a})
  expected = np.max(a, 1) 
  assert np.allclose(val, expected)

def test_neg_sequence_max():
  np.random.seed(0)
  a = np.float32(-np.random.rand(20,100,8))
  src = sequence.input(shape=(8), sequence_axis=Axis("Seq"))
  out = sequence.reduce_max(src)
  val = out.eval({src:a})
  expected = np.max(a, 1) 
  assert np.allclose(val, expected)

def np_softmax(a):
  m = np.max(a, 1, keepdims=True)
  e = np.exp((a-m))
  s = np.sum(e,1, keepdims=True)
  return e/s
  
def test_sequence_softmax():
  np.random.seed(0)
  a = np.float32(np.random.rand(20,100,8))
  src = sequence.input(shape=(8), sequence_axis=Axis("Seq"))
  out = sequence.softmax(src)
  val = out.eval({src:a})
  expected = np_softmax(a)
  assert np.allclose(val, expected)


def test_to_sequence_basic(device_id):
    dev = cntk_device(device_id)
    x = C.input((C.FreeDimension, 2))
    x_seq = C.to_sequence(x)
    assert len(x_seq.dynamic_axes) == 2

    x_data = np.asarray([[[1, 2], [-1000, -1000]], [[3, 4], [5, 6]]], dtype=np.float32)
    result = x_seq.eval({x : x_data}, device=dev)
    assert np.array_equal(result, x_data)

    x = C.input((C.FreeDimension, 2, 3), is_sparse=True)
    x_seq_lens = C.input(())
    x_seq = C.to_sequence(x, x_seq_lens)
    
    seq1_data = [[[0, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 0, 1]]]
    csr_seq1 = _to_csr(seq1_data)
    ndarrayview1 = C.NDArrayView.from_csr(csr_seq1, shape=(2, 2, 3), device=C.cpu())
    seq2_data = [[0, 1, 1], [1, 1, 0]]
    csr_seq2 = _to_csr([seq2_data, [[0, 0, 0], [0, 0, 0]]])
    ndarrayview2 = C.NDArrayView.from_csr(csr_seq2, shape=(2, 2, 3), device=C.cpu())

    x_data = C.Value.create(C.input((2, 2, 3), is_sparse=True), [ndarrayview1, ndarrayview2], device=dev).data
    x_seq_lens_data = np.asarray([2, 1], dtype=np.float32)
    result = x_seq.eval({x : x_data, x_seq_lens : x_seq_lens_data}, device=dev, as_numpy=False)
    result_dense = _to_dense(result, True)
    assert np.array_equal(result_dense[0], seq1_data)
    assert np.array_equal(result_dense[1], [seq2_data])


def test_to_sequence_backprop(device_id):
    dev = cntk_device(device_id)
    input_vocab_size=3
    emb_dim = 2
    hidden_dim = 2
    num_labels = 2
    x_seq_input = C.sequence.input(input_vocab_size, is_sparse=True, name='features')
    with C.default_options(initial_state=0.1):
        model = C.layers.Embedding(emb_dim, name='embed')(x_seq_input)
        model = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(model)
        model = C.layers.Dense(num_labels, name='classify')(model)

    z = model
    label_seq_input = C.sequence.input(num_labels, is_sparse=True, name='labels')
    ce = C.cross_entropy_with_softmax(z, label_seq_input)

    seq1_data = [[0, 1, 1], [0, 1, 0], [1, 0, 0]]
    seq2_data = [[0, 0, 1], [0, 1, 1]]
    seq1_label_data = [[0, 1], [0, 1], [1, 0]]
    seq2_label_data = [[1, 0], [0, 1]]
    label_seq_data = [_to_csr(seq1_label_data), _to_csr(seq2_label_data)]
    param_grads_1, loss_result_1 = ce.grad({x_seq_input : [_to_csr(seq1_data), _to_csr(seq2_data)], label_seq_input : label_seq_data},
                                           wrt=ce.parameters, outputs=[ce], as_numpy=False)
    
    # Create a clone of the model that uses a non-sequence input 
    # and converts it to a sequence using to_sequence
    x_non_seq_input = C.input((C.FreeDimension, input_vocab_size), is_sparse=True, name='non_seq_features')
    x_seq_lens = C.input((), name='sequence_lengths')
    x_seq = C.to_sequence(x_non_seq_input, x_seq_lens)
    x_seq = C.reconcile_dynamic_axes(C.times(x_seq, np.eye(input_vocab_size, dtype=np.float32)), label_seq_input)
    ce_clone = ce.clone('share', {x_seq_input : x_seq})

    x_non_seq_data = C.NDArrayView.from_csr(_to_csr([seq1_data, seq2_data + [[0, 0, 0]]]), shape=(2, 3, 3))
    x_seq_lens_data = np.asarray([3, 2], dtype=np.float32)

    x_non_seq_input = next(argument for argument in ce_clone.arguments if argument.name == 'non_seq_features')
    label_seq_input = next(argument for argument in ce_clone.arguments if argument.name == 'labels')
    x_seq_lens = next(argument for argument in ce_clone.arguments if argument.name == 'sequence_lengths')
    param_grads_2, loss_result_2 = ce_clone.grad({x_non_seq_input : x_non_seq_data, x_seq_lens : x_seq_lens_data, label_seq_input : label_seq_data},
                                                 wrt=ce_clone.parameters, outputs=[ce_clone], as_numpy=False)


    assert np.array_equal(loss_result_1.as_sequences()[0], loss_result_2.as_sequences()[0])
    assert np.array_equal(loss_result_1.as_sequences()[1], loss_result_2.as_sequences()[1])
    
    for param in param_grads_1:
        if not param_grads_1[param].is_sparse:
            reference_grad_value = param_grads_1[param].asarray()
            grad_value = param_grads_2[param].asarray()
            assert np.array_equal(reference_grad_value, grad_value)

