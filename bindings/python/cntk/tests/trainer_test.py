# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import numpy as np
from .. import Function
from ..ops import times
from ..ops.tests.ops_test_utils import cntk_device
from ..utils import one_hot
from ..trainer import *
from ..learner import *
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum
import pytest
from scipy.sparse import csr_matrix as csr

def test_trainer(tmpdir):
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, ce, errs,
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant, True)])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])

    p = str(tmpdir / 'checkpoint.dat')
    trainer.save_checkpoint(p)
    trainer.restore_from_checkpoint(p)

    assert trainer.model.name == 'z'

    # Ensure that Swig is not leaking raw types
    assert isinstance(trainer.model, Function)
    assert trainer.model.__doc__
    assert isinstance(trainer.parameter_learners[0], Learner)

def test_output_to_retain():
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, ce, errs,
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant, True)])
    in1_value = [[[1]], [[2]]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])
    assert np.allclose(var_map[z_output], np.asarray(in1_value)+20)

def test_eval_sparse_dense(tmpdir, device_id):
    from cntk import Axis
    from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
    from cntk.ops import input_variable, times

    input_vocab_dim = label_vocab_dim = 69

    ctf_data = '''\
0	|S0 3:1 |# <s>	|S1 3:1 |# <s>
0	|S0 4:1 |# A	|S1 32:1 |# ~AH
0	|S0 5:1 |# B	|S1 36:1 |# ~B
0	|S0 4:1 |# A	|S1 31:1 |# ~AE
0	|S0 7:1 |# D	|S1 38:1 |# ~D
0	|S0 12:1 |# I	|S1 47:1 |# ~IY
0	|S0 1:1 |# </s>	|S1 1:1 |# </s>
2	|S0 60:1 |# <s>	|S1 3:1 |# <s>
2	|S0 61:1 |# A	|S1 32:1 |# ~AH
'''
    ctf_file = str(tmpdir/'2seqtest.txt')
    with open(ctf_file, 'w') as f:
        f.write(ctf_data)

    mbs = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim,  is_sparse=True)
    )), randomize=False, epoch_size = 2)

    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    raw_input = input_variable(
        shape=input_vocab_dim, dynamic_axes=input_dynamic_axes,
        name='raw_input', is_sparse=True)

    mb_valid = mbs.next_minibatch(minibatch_size_in_samples=100,
            input_map={raw_input : mbs.streams.features},
            device=cntk_device(device_id))

    z = times(raw_input, np.eye(input_vocab_dim))
    e_reader = z.eval(mb_valid, device=cntk_device(device_id))

    # CSR with the raw_input encoding in ctf_data
    one_hot_data = [
            [3, 4, 5, 4, 7, 12, 1],
            [60, 61]
            ]
    data = [csr(np.eye(input_vocab_dim, dtype=np.float32)[d]) for d in
            one_hot_data]
    e_csr = z.eval({raw_input: data}, device=cntk_device(device_id))
    assert np.all([np.allclose(a, b) for a,b in zip(e_reader, e_csr)])

    # One-hot with the raw_input encoding in ctf_data
    data = one_hot(one_hot_data, num_classes=input_vocab_dim, device=cntk_device(device_id))
    e_hot = z.eval({raw_input: data}, device=cntk_device(device_id))
    assert np.all([np.allclose(a, b) for a,b in zip(e_reader, e_hot)])

@pytest.mark.parametrize("batch_index_data", [
     [2,3],
     [0,1,6],
    ])
def test_eval_sparse_no_seq(batch_index_data, device_id):
    dim = 10
    multiplier = 2
    for var_is_sparse in [True, False]:
        in1 = input_variable(shape=(dim,), is_sparse=var_is_sparse)
        z = times(in1, multiplier*np.eye(dim))
        batch = np.eye(dim)[batch_index_data]
        expected = batch * multiplier
        sparse_val = csr(batch.astype('f'))
        result = z.eval({in1: [sparse_val]}, device=cntk_device(device_id))
        assert np.allclose(result, [expected])

@pytest.mark.parametrize("batch", [
    [csr([0,1,2,0])],
    [
        csr([[0, 2, 0, 7], [10, 20, 0, 0]]),
        csr([0, 0, 0, 3])
    ]
    ])
def test_eval_sparse_seq_1(batch, device_id):
    dim = 4
    multiplier = 2
    for var_is_sparse in [True, False]:
        in1 = input_variable(shape=(dim,), is_sparse=var_is_sparse)
        z = times(in1, multiplier*np.eye(dim))
        if isinstance(batch[0], list):
            expected = [np.vstack([m.todense() * multiplier for m in seq]) for seq in
                    batch]
        else:
            expected = [seq.todense() * multiplier for seq in batch]
        result = z.eval({in1: batch}, device=cntk_device(device_id))

        assert np.all([np.allclose(a,b) for a,b in zip(result, expected)]), \
                "%s != %s"%(result,expected)


@pytest.mark.parametrize("one_hot_batch", [
     ([[2,5],
      [0,1,6]]),
     ([[1],[1],[2],[3]]),
     ([[1,5],
         [4]]),
    ])
def test_eval_one_hot_seq(one_hot_batch, device_id):
    dim = 10
    multiplier = 2

    for var_is_sparse in [True, False]:
        in1 = input_variable(shape=(dim,), is_sparse=var_is_sparse)
        # Convert CNTK node value to dense so that we can compare it later
        z = times(in1, np.eye(dim)*multiplier)
        # Convert expectation to dense
        expected = [np.eye(dim)[seq]*multiplier for seq in one_hot_batch]
        batch = one_hot(one_hot_batch, num_classes=dim, device=cntk_device(device_id))
        result = z.eval({in1: batch}, device=cntk_device(device_id))
        assert np.all([np.allclose(a,b) for a,b in zip(result, expected)])

@pytest.mark.parametrize("one_hot_batch, dim", [
    ([[11]], 10),
    ([[0, 1]], 1),
    ])
def test_eval_one_hot_bad(one_hot_batch, dim, device_id):
    with pytest.raises(ValueError):
        batch = one_hot(one_hot_batch, num_classes=dim, device=cntk_device(device_id))

