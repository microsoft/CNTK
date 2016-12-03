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
        
def create_data_parallel_distributed_trainer(quantized, warm_start):
    return distributed.data_parallel_distributed_trainer(
        use_async_buffered_parameter_update=False,
        num_quantization_bits=(1 if quantized else 32),
        distributed_after=warm_start)

def create_block_momentum_distributed_trainer(quantized, warm_start):
    return distributed.block_momentum_distributed_trainer(
        block_size=1024,
        distributed_after = warm_start)

def create_block_momentum_distributed_trainer_with_time_constant(quantized, warm_start):
    return distributed.block_momentum_distributed_trainer(
        block_size=1024,
        block_momentum_as_time_constant=4096,
        distributed_after=warm_start)

def run_distributed_trainer(tmpdir, quantized, create_func):

    in1 = input_variable(shape=1)
    labels = input_variable(shape=1)
    p = parameter(shape=2, init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    warm_start = (100 if quantized else 0)

    dist_trainer = create_func(quantized, warm_start)

    assert dist_trainer.distributed_after == warm_start

    communicator = dist_trainer.communicator()
    workers = communicator.workers()
    current_worker = communicator.current_worker()
    found_rank = False
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            found_rank = True

    assert found_rank

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, ce, errs, \
            momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant),
            distributed_trainer=dist_trainer)
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
    assert isinstance(trainer.parameter_learners[0], Learner)

def test_distributed(tmpdir, is_1bit_sgd):
    run_distributed_trainer(tmpdir, quantized=(True if is_1bit_sgd==1 else False), create_func=create_data_parallel_distributed_trainer)
    if is_1bit_sgd == 1:
        run_distributed_trainer(tmpdir, True, create_func=create_block_momentum_distributed_trainer)
        run_distributed_trainer(tmpdir, True,  create_func=create_block_momentum_distributed_trainer_with_time_constant)
    distributed.Communicator.finalize()
    