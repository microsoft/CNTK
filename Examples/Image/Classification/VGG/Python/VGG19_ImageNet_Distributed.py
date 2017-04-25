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

from cntk.logging import *
from cntk.ops import *
from cntk.distributed import data_parallel_distributed_learner, Communicator
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.layers import Placeholder, Block, Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential, For
from cntk.initializer import normal
from cntk.train.training_session import *

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 1000
model_name   = "VGG19.model"

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            ImageDeserializer.crop(crop_type='randomside', side_ratio='0.4375:0.875', jitter_type='uniratio') # train uses jitter
        ]
    else: 
        transforms += [
            ImageDeserializer.crop(crop_type='center', side_ratio=0.5833333) # test has no jitter
        ]

    transforms += [
        ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize = is_training, 
        max_samples=total_number_of_samples,
        multithreaded_deserializer = True)

# Create the network.
def create_vgg19():

    # Input variables denoting the features and label data
    feature_var = input((num_channels, image_height, image_width))
    label_var = input((num_classes))

    # apply model to input
    # remove mean value 
    input = minus(feature_var, constant([[[104]], [[117]], [[124]]]), name='mean_removed_input')
    
    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
            For(range(2), lambda i: [
                Convolution2D((3,3), 64, name='conv1_{}'.format(i)), 
                Activation(activation=relu, name='relu1_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool1'),

            For(range(2), lambda i: [
                Convolution2D((3,3), 128, name='conv2_{}'.format(i)), 
                Activation(activation=relu, name='relu2_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool2'),

            For(range(4), lambda i: [
                Convolution2D((3,3), 256, name='conv3_{}'.format(i)), 
                Activation(activation=relu, name='relu3_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool3'),

            For(range(4), lambda i: [
                Convolution2D((3,3), 512, name='conv4_{}'.format(i)), 
                Activation(activation=relu, name='relu4_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool4'),

            For(range(4), lambda i: [
                Convolution2D((3,3), 512, name='conv5_{}'.format(i)), 
                Activation(activation=relu, name='relu5_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool5'),

            Dense(4096, name='fc6'), 
            Activation(activation=relu, name='relu6'), 
            Dropout(0.5, name='drop6'), 
            Dense(4096, name='fc7'), 
            Activation(activation=relu, name='relu7'), 
            Dropout(0.5, name='drop7'),
            Dense(num_classes, name='fc8')
            ])(input)

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)
    pe5 = classification_error(z, label_var, topN=5)

    log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'pe5': pe5, 
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, progress_printer):
    # Set learning parameters
    lr_per_mb         = [0.01]*20 + [0.001]*20 + [0.0001]*20 + [0.00001]*10 + [0.000001]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_mb, unit=cntk.learner.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = cntk.learner.momentum_schedule(0.9)
    l2_reg_weight     = 0.0005 # CNTK L2 regularization is per sample, thus same as Caffe
    
    # Create learner
    local_learner = cntk.learner.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=l2_reg_weight)
    # Since we reuse parameter settings (learning rate, momentum) from Caffe, we set unit_gain to False to ensure consistency 
    parameter_learner = data_parallel_distributed_learner(
        local_learner, 
        num_quantization_bits=num_quantization_bits,
        distributed_after=0)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_printer)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Train all minibatches 
    training_session(
        trainer=trainer, mb_source = train_source, 
        model_inputs_to_streams = input_map, 
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(filename = os.path.join(model_path, model_name), restore=restore),
        test_config = TestConfig(source=test_source, mb_size=minibatch_size)
    ).train()

# Train and evaluate the network.
def vgg19_train_and_eval(train_data, test_data, num_quantization_bits=32, minibatch_size=128, epoch_size = 1281167, max_epochs=80, 
                         restore=True, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False):
    _cntk_py.set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_vgg19()
    trainer = create_trainer(network, epoch_size, num_quantization_bits, progress_printer)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore)
 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='80')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='128')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='1281167')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'val_map.txt')

    try:
        vgg19_train_and_eval(train_data, test_data, 
                             minibatch_size=args['minibatch_size'], 
                             epoch_size=args['epoch_size'],
                             num_quantization_bits=args['quantized_bits'],
                             max_epochs=args['num_epochs'],
                             restore=not args['restart'],
                             log_to_file=args['logdir'],
                             num_mbs_per_log=200,
                             gen_heartbeat=True)
    finally:
        cntk.distributed.Communicator.finalize()    
