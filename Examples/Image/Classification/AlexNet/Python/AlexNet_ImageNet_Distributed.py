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
from cntk.train.training_session import *
from cntk import *
from cntk.train.distributed import *
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
import cntk.io.transforms as xforms
from cntk.layers import Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential
from cntk.initializer import normal

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 227
image_width  = 227
num_channels = 3  # RGB
num_classes  = 1000
model_name   = "AlexNet.model"

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.88671875, jitter_type='uniratio') # train uses jitter
        ]
    else:
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.88671875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize = is_training,
        max_samples=total_number_of_samples,
        multithreaded_deserializer = True)

# Local Response Normalization layer. See Section 3.3 of the paper:
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# The mathematical equation is:
#   b_{x,y}^i=a_{x,y}^i/(k+\alpha\sum_{j=max(0,i-n)}^{min(N-1, i+n)}(a_{x,y}^j)^2)^\beta
# where a_{x,y}^i is the activity of a neuron comoputed by applying kernel i at position (x,y)
# N is the total number of kernals, n is half normalization width.
def LocalResponseNormalization(k, n, alpha, beta, name=''):
    x = cntk.placeholder(name='lrn_arg')
    x2 = cntk.square(x)
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
    x2s = cntk.reshape(x2, (1, cntk.InferredDimension), 0, 1)
    W = cntk.constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = cntk.convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = cntk.reshape(y, cntk.InferredDimension, 0, 2)
    den = cntk.exp(beta * cntk.log(k + b))
    apply_x = cntk.element_divide(x, den)
    return apply_x

# Create the network.
def create_alexnet():

    # Input variables denoting the features and label data
    feature_var = input((num_channels, image_height, image_width))
    label_var = input((num_classes))

    # apply model to input
    # remove mean value 
    mean_removed_features = minus(feature_var, constant(114), name='mean_removed_input')
    
    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
            Convolution2D((11,11), 96, init=normal(0.01), pad=False, strides=(4,4), name='conv1'),
            Activation(activation=relu, name='relu1'),
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm1'),
            MaxPooling((3,3), (2,2), name='pool1'),

            Convolution2D((5,5), 192, init=normal(0.01), init_bias=0.1, name='conv2'), 
            Activation(activation=relu, name='relu2'),
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm2'),
            MaxPooling((3,3), (2,2), name='pool2'),

            Convolution2D((3,3), 384, init=normal(0.01), name='conv3'), 
            Activation(activation=relu, name='relu3'),
            Convolution2D((3,3), 384, init=normal(0.01), init_bias=0.1, name='conv4'), 
            Activation(activation=relu, name='relu4'),
            Convolution2D((3,3), 256, init=normal(0.01), init_bias=0.1, name='conv5'), 
            Activation(activation=relu, name='relu5'), 
            MaxPooling((3,3), (2,2), name='pool5'), 

            Dense(4096, init=normal(0.005), init_bias=0.1, name='fc6'),
            Activation(activation=relu, name='relu6'),
            Dropout(0.5, name='drop6'),
            Dense(4096, init=normal(0.005), init_bias=0.1, name='fc7'),
            Activation(activation=relu, name='relu7'),
            Dropout(0.5, name='drop7'),
            Dense(num_classes, init=normal(0.01), name='fc8')
            ])(mean_removed_features)

    # loss and metric
    ce  = cross_entropy_with_softmax(z, label_var)
    pe  = classification_error(z, label_var)
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
def create_trainer(network, epoch_size, num_quantization_bits, printer, block_size, warm_up):
    # Set learning parameters
    lr_per_mb         = [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_mb, unit=cntk.learners.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = cntk.learners.momentum_schedule(0.9)
    l2_reg_weight     = 0.0005 # CNTK L2 regularization is per sample, thus same as Caffe

    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    # Create learner
    local_learner = cntk.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=l2_reg_weight)
    # Since we reuse parameter settings (learning rate, momentum) from Caffe, we set unit_gain to False to ensure consistency

    # Create trainer
    if block_size != None:
        parameter_learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    return cntk.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, printer)

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
        checkpoint_config = CheckpointConfig(filename=os.path.join(model_path, model_name), restore=restore),
        test_config= TestConfig(source=test_source, mb_size=minibatch_size)
    ).train()

# Train and evaluate the network.
def alexnet_train_and_eval(train_data, test_data, num_quantization_bits=32, block_size=3200, warm_up=0, minibatch_size=256, epoch_size = 1281167, max_epochs=112,
                           restore=True, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=True):
    _cntk_py.set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_alexnet()
    trainer = create_trainer(network, epoch_size, num_quantization_bits, progress_printer, block_size, warm_up)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='112')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='256')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='1281167')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
        if Communicator.rank() == 0:
            cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
        else:
            cntk.device.try_set_default_device(cntk.device.cpu())

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')

    try:
        alexnet_train_and_eval(train_data, test_data,
                               max_epochs=args['num_epochs'],
                               restore=not args['restart'],
                               log_to_file=args['logdir'],
                               num_mbs_per_log=200,
                               num_quantization_bits=args['quantized_bits'],
                               block_size=args['block_samples'],
                               warm_up=args['distributed_after'],
                               minibatch_size=args['minibatch_size'],
                               epoch_size=args['epoch_size'],
                               gen_heartbeat=True)
    finally:
        cntk.train.distributed.Communicator.finalize()
