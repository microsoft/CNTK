# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
import argparse
import _cntk_py
import cntk

from cntk import Trainer
from cntk.train.distributed import Communicator, data_parallel_distributed_learner, block_momentum_distributed_learner
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learners import fsadagrad, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.train.training_session import *
from cntk.logging import *

abs_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

default_quantization_bits = 32

# model dimensions
input_vocab_dim  = 69
label_vocab_dim  = 69

use_attention = True

def create_reader(path, randomize, size=INFINITELY_REPEAT):
    if not os.path.exists(path):
        raise RuntimeError("File '%s' does not exist." % (path))

    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_dim,  is_sparse=True)
    )), randomize=randomize, max_samples = size)

def train_and_test(s2smodel, train_reader, test_reader, block_size, num_quantization_bits, max_epochs, epoch_size, minibatch_size, progress_printer, warm_up):
    from Sequence2Sequence import create_criterion_function, create_model_train
    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)

    # Create learner
    if block_size is not None and num_quantization_bits != default_quantization_bits:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    lr = 0.001 if use_attention else 0.005   # TODO: can we use the same value for both?
    local_learner = fsadagrad(model_train.parameters,
                        lr       = learning_rate_schedule([lr]*2+[lr/2]*3+[lr/4], UnitType.sample, epoch_size),
                        momentum = momentum_as_time_constant_schedule(1100),
                        gradient_clipping_threshold_per_sample=2.3,
                        gradient_clipping_with_truncation=True)

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    trainer = Trainer(None, criterion, learner, progress_printer)

    train_bind = {criterion.arguments[0]: train_reader.streams.features,
                  criterion.arguments[1]: train_reader.streams.labels}

    training_session(
        mb_source = train_reader,
        trainer=trainer,
        model_inputs_to_streams=train_bind,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=CheckpointConfig(frequency = epoch_size,
                                           filename = os.path.join(model_path, "SequenceToSequence"),
                                           restore = False),
        cv_config=CrossValidationConfig(source=test_reader, mb_size=minibatch_size)
    ).train()

def sequence_to_sequence_translator(train_data, test_data, epoch_size=908241, num_quantization_bits=default_quantization_bits, block_size=3200, warm_up=0, minibatch_size=72, max_epochs=10, randomize_data=False, log_to_file=None, num_mbs_per_log=10, gen_heartbeat=False):
    cntk.debugging.set_computation_network_trace_level(0)
    from _cntk_py import set_fixed_random_seed
    set_fixed_random_seed(1)

    from Sequence2Sequence import create_model

    distributed_sync_report_freq = None
    if block_size is not None:
        distributed_sync_report_freq = 1

    progress_printer = ProgressPrinter(freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs,
        distributed_freq=distributed_sync_report_freq)

    # create inputs and create model
    model = create_model()

    train_reader = create_reader(train_data, randomize_data, size=max_epochs*epoch_size)
    test_reader = create_reader(test_data, False, size=max_epochs*epoch_size*10)

    train_and_test(model, train_reader, test_reader, block_size, num_quantization_bits, max_epochs, epoch_size, minibatch_size, progress_printer, warm_up)

if __name__ == '__main__':
    data_path  = os.path.join(abs_path, "..", "Data")

    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CMUDict dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-ms', '--minibatch_size', help='Minibatch size', type=int, required=False, default='16')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-es', '--epoch_size', help='Epoch size', type=int, required=False, default='64')
    parser.add_argument('-r', '--randomize_data', help='Randomize training data', type=bool, required=False, default=False)
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    train_data = os.path.join(data_path, 'cmudict-0.7b.train-dev-20-21.ctf')
    test_data = os.path.join(data_path, 'cmudict-0.7b.test.ctf')

    try:
        sequence_to_sequence_translator(train_data, test_data,
                                        epoch_size=args['epoch_size'],
                                        num_quantization_bits=args['quantized_bits'],
                                        block_size=args['block_samples'],
                                        warm_up=args['distributed_after'],
                                        minibatch_size= args['minibatch_size'],
                                        max_epochs=args['epochs'],
                                        randomize_data=args['randomize_data'],
                                        log_to_file=args['logdir'],
                                        num_mbs_per_log=10)

    finally:
        Communicator.finalize()
