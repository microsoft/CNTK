# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import argparse
import numpy as np
import cntk
import _cntk_py
from cntk.train.distributed import *
from cntk.io import MinibatchSource, HTKFeatureDeserializer, HTKMLFDeserializer, StreamDef, StreamDefs
from cntk.layers import Recurrence, Dense, LSTM
from cntk.learners import *
from cntk.layers.models import Sequential, For
from cntk import input, cross_entropy_with_softmax, classification_error, sequence
from cntk.train.training_session import *

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

feature_dim = 33
num_classes = 132
context = 2

# Create a minibatch source.
def create_mb_source(features_file, labels_file, label_mapping_filem, total_number_of_samples):
    for file_name in [features_file, labels_file, label_mapping_file]:
        if not os.path.exists(file_name):
            raise RuntimeError("File '%s' does not exist. Please check that datadir argument is set correctly." % (file_name))

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    ld = HTKMLFDeserializer(label_mapping_file, StreamDefs(
        awesome_labels = StreamDef(shape=num_classes, mlf=labels_file)))

    # Enabling BPTT with truncated_length > 0
    return MinibatchSource([fd,ld], truncation_length=250, epoch_size=total_number_of_samples)

def create_recurrent_network():
    # Input variables denoting the features and label data
    features = sequence.input(((2*context+1)*feature_dim))
    labels = sequence.input((num_classes))

    # create network
    model = Sequential([For(range(3), lambda : Recurrence(LSTM(256))),
                        Dense(num_classes)])
    z = model(features)
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error    (z, labels)

    return {
        'feature': features,
        'label': labels,
        'ce' : ce,
        'errs' : errs,
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers):
    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    lr = [0.001]

    local_learner = fsadagrad(network['output'].parameters,
                              lr=learning_rate_schedule(lr, UnitType.sample, epoch_size),
                              momentum=momentum_as_time_constant_schedule(1000),
                              gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    if block_size != None:
        parameter_learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['errs']), parameter_learner, progress_writers)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size):
    input_map = {
        network['feature']: train_source.streams.amazing_features,
        network['label']: train_source.streams.awesome_labels
    }

    training_session(
        trainer=trainer,
        mb_source = train_source,
        var_to_stream = input_map,
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(frequency = epoch_size,
                                             filename = os.path.join(model_path, "HKT_LSTM_Truncated"),
                                             restore = False),
        cv_config = CrossValidationConfig(source=test_source, mb_size=minibatch_size)
    ).train()

def htk_lstm_truncated(features_file, labels_file, label_mapping_file, minibatch_size=64, epoch_size=640000, num_quantization_bits=32,
                            block_size=3200, warm_up=0, max_epochs=5, num_mbs_per_log=None, gen_heartbeat=False,log_to_file=None, tensorboard_logdir=None):

    cntk.debugging.set_computation_network_trace_level(0)

    network = create_recurrent_network()

    progress_writers = [cntk.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)]

    if tensorboard_logdir is not None:
        progress_writers.append(cntk.logging.TensorBoardProgressWriter(
        freq=num_mbs_per_log,
            log_dir=tensorboard_logdir,
        rank=Communicator.rank(),
            model=network['output']))

    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers)
    train_source = create_mb_source(features_file, labels_file, label_mapping_file, total_number_of_samples=max_epochs * epoch_size)
    # Testing with training data, just for testing purposes
    test_source = create_mb_source(features_file, labels_file, label_mapping_file, total_number_of_samples=max_epochs * epoch_size)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    data_path  = os.path.join(abs_path, "..", "Data")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the AN4 files are located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='64')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='50000')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    os.chdir(data_path)

    features_file = os.path.join(data_path, 'glob_0000.scp')
    labels_file = os.path.join(data_path, 'glob_0000.mlf')
    label_mapping_file = os.path.join(data_path, 'state.list')

    try:
        htk_lstm_truncated(features_file, labels_file, label_mapping_file,
                                minibatch_size=args['minibatch_size'],
                                epoch_size=args['epoch_size'],
                                num_quantization_bits=args['quantized_bits'],
                                block_size=args['block_samples'],
                                warm_up=args['distributed_after'],
                                max_epochs=args['num_epochs'],
                                log_to_file=args['logdir'],
                                num_mbs_per_log=100,
                                gen_heartbeat=False,
                                tensorboard_logdir=args['tensorboard_logdir'])
    finally:
        os.chdir(abs_path)
        Communicator.finalize()
