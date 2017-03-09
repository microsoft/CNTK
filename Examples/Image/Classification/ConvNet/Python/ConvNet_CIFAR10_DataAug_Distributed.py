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
import cntk.io.transforms as xforms

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

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
    return cntk.io.MinibatchSource(
        cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
            features = cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = cntk.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=train,
        epoch_size=total_number_of_samples,
        multithreaded_deserializer = True)

# Create the network.
def create_conv_network():

    # Input variables denoting the features and label data
    feature_var = cntk.ops.input_variable((num_channels, image_height, image_width))
    label_var = cntk.ops.input_variable((num_classes))

    # apply model to input
    scaled_input = cntk.ops.element_times(cntk.ops.constant(0.00390625), feature_var)

    with cntk.layers.default_options(activation=cntk.ops.relu, pad=True):
        z = cntk.models.Sequential([
            cntk.models.For(range(2), lambda : [
                cntk.layers.Convolution2D((3,3), 64),
                cntk.layers.Convolution2D((3,3), 64),
                cntk.layers.MaxPooling((3,3), (2,2))
            ]),
            cntk.models.For(range(2), lambda i: [
                cntk.layers.Dense([256,128][i]),
                cntk.layers.Dropout(0.5)
            ]),
            cntk.layers.Dense(num_classes, activation=None)
        ])(scaled_input)

    # loss and metric
    ce = cntk.ops.cross_entropy_with_softmax(z, label_var)
    pe = cntk.ops.classification_error(z, label_var)

    cntk.utils.log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }


# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up):
    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learner.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant  = [0]*20 + [600]*20 + [1200]
    mm_schedule       = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight     = 0.002

    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = cntk.learner.momentum_sgd(network['output'].parameters,
                                              lr_schedule, mm_schedule,
                                              l2_regularization_weight=l2_reg_weight)

    if block_size != None:
        parameter_learner = cntk.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = cntk.distributed.data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner)

# Train and test
def train_and_test(network, trainer, train_source, test_source, progress_writers, minibatch_size, epoch_size, restore, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    training_session = cntk.training_session(
        training_minibatch_source = train_source,
        trainer = trainer,
        model_inputs_to_mb_source_mapping = input_map,
        mb_size_schedule = cntk.minibatch_size_schedule(minibatch_size),
        progress_printer = progress_writers,
        checkpoint_frequency = epoch_size,
        checkpoint_filename = os.path.join(model_path, "ConvNet_CIFAR10_DataAug"),
#        save_all_checkpoints = False,
        progress_frequency=epoch_size,
        cv_source = test_source,
        cv_mb_size_schedule=cntk.minibatch_size_schedule(minibatch_size),
#        cv_frequency = epoch_size,
        restore=restore)

    # Train all minibatches
    if profiling:
        cntk.start_profiler(sync_gpu=True)

    training_session.train()

    if profiling:
        cntk.stop_profiler()

# Train and evaluate the network.
def convnet_cifar10_dataaug(train_data, test_data, mean_data, minibatch_size=64, epoch_size=50000, num_quantization_bits=32,
                            block_size=3200, warm_up=0, max_epochs=2, restore=False, log_to_file=None, 
                            num_mbs_per_log=None, gen_heartbeat=False, profiling=False, tensorboard_logdir=None):
    _cntk_py.set_computation_network_trace_level(0)

    network = create_conv_network()

    progress_printer = cntk.utils.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=cntk.distributed.Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    tensorboard_writer = cntk.utils.TensorBoardProgressWriter(
        freq=num_mbs_per_log,
        log_dir=tensorboard_logdir if tensorboard_logdir is not None else 'log',
        rank=cntk.distributed.Communicator.rank(),
        model=network['output'])

    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up)
    train_source = create_image_mb_source(train_data, mean_data, train=True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, train=False, total_number_of_samples=cntk.io.FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, [progress_printer, tensorboard_writer], minibatch_size,
                   epoch_size, restore, profiling)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where to tensorboard logs should be written', required=False, default='log')
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='64')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='50000')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
    if args['device'] is not None:
        cntk.device.set_default_device(cntk.device.gpu(args['device']))

    mean_data=os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    try:
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
                                gen_heartbeat=False,
                                profiling=args['profile'],
                                tensorboard_logdir=args['tensorboard_logdir'])
    finally:
        cntk.distributed.Communicator.finalize()

