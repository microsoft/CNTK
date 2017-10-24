# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import argparse
import numpy as np
import cntk as C
import _cntk_py
import cntk.io.transforms as xforms
from cntk.train.training_session import *
from cntk.logging import *
from cntk.debugging import *

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
model_path = os.path.join(abs_path, "Models")

from ConvNet_CIFAR10_DataAug import create_convnet_cifar10_model

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

# Create a minibatch source.
def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(
            map_file, 
            C.io.StreamDefs(features=C.io.StreamDef(field='image', transforms=transforms), # 1st col in mapfile referred to as 'image'
                            labels=C.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)

# Create the network.
def create_conv_network():
    # Input variables denoting the features and label data
    feature_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # apply model to input
    scaled_input = C.element_times(C.constant(0.00390625), feature_var)

    z = create_convnet_cifar10_model(num_classes)(scaled_input)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    C.logging.log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers):
    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule       = C.learning_rate_schedule(lr_per_sample, unit=C.learners.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant  = [0]*20 + [600]*20 + [1200]
    mm_schedule       = C.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight     = 0.002

    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = C.learners.momentum_sgd(network['output'].parameters,
                                            lr_schedule, mm_schedule,
                                            l2_regularization_weight=l2_reg_weight)

    if block_size != None:
        parameter_learner = C.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = C.train.distributed.data_parallel_distributed_learner(local_learner, 
                                                                                  num_quantization_bits=num_quantization_bits, 
                                                                                  distributed_after=warm_up)

    # Create trainer
    return C.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_writers)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Train all minibatches
    if profiling:
        start_profiler(sync_gpu=True)

    training_session(
        trainer=trainer, mb_source = train_source,
        model_inputs_to_streams = input_map, 
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(frequency = epoch_size,
                                             filename = os.path.join(model_path, "ConvNet_CIFAR10_DataAug"),
                                             restore = restore),
        test_config = TestConfig(test_source, minibatch_size=minibatch_size)
    ).train()

    if profiling:
        stop_profiler()

# Train and evaluate the network.
def convnet_cifar10_dataaug(train_data, test_data, mean_data, minibatch_size=64, epoch_size=50000, num_quantization_bits=32,
                            block_size=3200, warm_up=0, max_epochs=2, restore=False, log_to_file=None, 
                            num_mbs_per_log=None, gen_heartbeat=False, profiling=False, tensorboard_logdir=None):
    _cntk_py.set_computation_network_trace_level(0)

    network = create_conv_network()

    distributed_sync_report_freq = None
    if block_size is not None:
        distributed_sync_report_freq = 1

    progress_writers = [C.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=C.train.distributed.Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs,
        distributed_freq=distributed_sync_report_freq)]

    if tensorboard_logdir is not None:
        progress_writers.append(C.logging.TensorBoardProgressWriter(
        freq=num_mbs_per_log,
        log_dir=tensorboard_logdir,
        rank=C.train.distributed.Communicator.rank(),
        model=network['output']))

    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers)
    train_source = create_image_mb_source(train_data, mean_data, train=True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located', 
                        required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', 
                        required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='64')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='50000')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, 
                        required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, 
                        required=False, default='0')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", 
                        required=False, default=None)
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', 
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", 
                        required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        C.device.try_set_default_device(C.device.gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    mean_data=os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    convnet_cifar10_dataaug(train_data, test_data, mean_data,
                            minibatch_size=args['minibatch_size'],
                            epoch_size=args['epoch_size'],
                            num_quantization_bits=args['quantized_bits'],
                            block_size=args['block_samples'],
                            warm_up=args['distributed_after'],
                            max_epochs=args['num_epochs'],
                            restore=not args['restart'],
                            log_to_file=args['logdir'],
                            num_mbs_per_log=100,
                            gen_heartbeat=True,
                            profiling=args['profile'],
                            tensorboard_logdir=args['tensorboard_logdir'])
    # Must call MPI finalize when process exit without exceptions
    C.train.distributed.Communicator.finalize()

