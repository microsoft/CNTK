# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
import numpy as np
import pytest
from .. import Function
from ..trainer import *
from ..learner import *
from .. import distributed
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum

def create_data_parallel_distributed_learner(learner, quantized, distributed_after):
    return distributed.data_parallel_distributed_learner(
        learner=learner,
        distributed_after=distributed_after,
        use_async_buffered_parameter_update=False,
        num_quantization_bits=(1 if quantized else 32))

def create_block_momentum_distributed_learner(learner, distributed_after):
    return distributed.block_momentum_distributed_learner(
        learner=learner,
        block_size=1024,
        distributed_after=distributed_after)

def create_block_momentum_distributed_learner_with_time_constant(learner, distributed_after):
    return distributed.block_momentum_distributed_learner(
        learner=learner,
        block_size=1024,
        block_momentum_as_time_constant=4096,
        distributed_after=distributed_after)

def run_distributed_training(tmpdir, create_func):

    in1 = input_variable(shape=1)
    labels = input_variable(shape=1)
    p = parameter(shape=2, init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    dist_learner = create_func(momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant, True))

    communicator = dist_learner.communicator()
    workers = communicator.workers()
    current_worker = communicator.current_worker()
    found_rank = False
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            found_rank = True

    assert found_rank

    trainer = Trainer(z, (ce, errs), [ dist_learner ])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])
    
    p = str(tmpdir / 'checkpoint.dat')
    trainer.save_checkpoint(p)
    trainer.restore_from_checkpoint(p)

    communicator.barrier()

    assert trainer.model.name == 'z'

    # Ensure that Swig is not leaking raw types
    assert isinstance(trainer.model, Function)
    assert trainer.model.__doc__


def test_distributed_mb_source(tmpdir):
    input_dim = 69

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
2	|S0 61:1 |# A	|S1 32:1 |# ~AH
3	|S0 60:1 |# <s>	|S1 3:1 |# <s>
3	|S0 61:1 |# A	|S1 32:1 |# ~AH
3	|S0 61:1 |# A	|S1 32:1 |# ~AH
3	|S0 61:1 |# A	|S1 32:1 |# ~AH
4	|S0 60:1 |# <s>	|S1 3:1 |# <s>
5	|S0 60:1 |# <s>	|S1 3:1 |# <s>
5	|S0 61:1 |# A	|S1 32:1 |# ~AH
6	|S0 60:1 |# <s>	|S1 3:1 |# <s>
6	|S0 61:1 |# A	|S1 32:1 |# ~AH
7	|S0 60:1 |# <s>	|S1 3:1 |# <s>
8	|S0 60:1 |# <s>	|S1 3:1 |# <s>
8	|S0 61:1 |# A	|S1 32:1 |# ~AH
9	|S0 60:1 |# <s>	|S1 3:1 |# <s>
9	|S0 61:1 |# A	|S1 32:1 |# ~AH
10	|S0 61:1 |# A	|S1 32:1 |# ~AH
'''
    from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, FULL_DATA_SWEEP

    ctf_file = str(tmpdir/'2seqtest.txt')
    with open(ctf_file, 'w') as f:
        f.write(ctf_data)

    # No randomization

    mb0 = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
        )), 
        randomize=False, epoch_size=36) # A bit more than a sweep
    mb1 = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
        )), 
        randomize=False, epoch_size=36) # A bit more than a sweep
    input = input_variable(shape=(input_dim,))
    label = input_variable(shape=(input_dim,))
    input_map = {
        input : mb0.streams.features,
        label : mb0.streams.labels
    }

    # Because we emulating two workers here, the minibatch_size_in_samples will be splitted in 2,
    # so below we expect 5 samples per worker.
    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 7) # Sequence 0

    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 4) # Sequence 3

    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 5) # Sequences 5, 7, 9

    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 7) # Sequence 0

    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 4) # Sequence 3

    data = mb0.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(len(data) == 0) # No data

    data = mb1.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=1)
    assert(data[input].num_samples == 4) # Sequences 2, 4

    data = mb1.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=1)
    assert(data[input].num_samples == 5) # Sequences 6, 8, 10

    data = mb1.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=1)
    assert(data[input].num_samples == 3) # Sequences 2

    data = mb1.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=1)
    assert(len(data) == 0) # No data

    # Radomization

    mb3 = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
        )), 
        randomize=True, epoch_size=FULL_DATA_SWEEP)

    mb4 = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
        )), 
        randomize=True, epoch_size=FULL_DATA_SWEEP)

    data = mb3.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 5)

    data = mb3.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 4)

    data = mb3.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 4)

    data = mb3.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 5)

    data = mb3.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=0)
    assert(data[input].num_samples == 7)

    data = mb4.next_minibatch(minibatch_size_in_samples=10, input_map=input_map, num_data_partitions=2, partition_index=1)
    assert(len(data) == 0) # Due to chunking we do not expect any data for rank 1


def test_distributed(tmpdir, is_1bit_sgd):
    quantized=(True if is_1bit_sgd==1 else False)

    simple_aggregation=lambda learner: create_data_parallel_distributed_learner(learner, False, 0)
    run_distributed_training(tmpdir, create_func=simple_aggregation)

    if is_1bit_sgd == 1:
        quantized_aggregation=lambda learner: create_data_parallel_distributed_learner(learner, True, 100)
        run_distributed_training(tmpdir, create_func=quantized_aggregation)

        block_momentum=lambda learner: create_block_momentum_distributed_learner(learner, 100)
        run_distributed_training(tmpdir, create_func=block_momentum)

        block_momentum_with_time=lambda learner: create_block_momentum_distributed_learner_with_time_constant(learner, 100)
        run_distributed_training(tmpdir, create_func=block_momentum_with_time)
    distributed.Communicator.finalize()
