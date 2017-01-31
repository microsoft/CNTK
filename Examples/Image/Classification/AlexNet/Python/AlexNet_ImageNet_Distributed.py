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

from cntk.utils import *
from cntk.ops import *
from cntk.distributed import data_parallel_distributed_learner, Communicator
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.blocks import Placeholder, Block
from cntk.layers import Convolution, Activation, MaxPooling, Dense, Dropout, default_options
from cntk.models import Sequential
from cntk.initializer import normal

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 227
image_width  = 227
num_channels = 3  # RGB
num_classes  = 1000

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            ImageDeserializer.crop(crop_type='randomside', side_ratio=0.88671875, jitter_type='uniratio') # train uses jitter
        ]
    else: 
        transforms += [
            ImageDeserializer.crop(crop_type='center', side_ratio=0.88671875) # test has no jitter
        ]

    transforms += [
        ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        epoch_size=total_number_of_samples,
        multithreaded_deserializer = True)

# Local Response Normalization layer. See Section 3.3 of the paper: 
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf 
# The mathematical equation is: 
#   b_{x,y}^i=a_{x,y}^i/(k+\alpha\sum_{j=max(0,i-n)}^{min(N-1, i+n)}(a_{x,y}^j)^2)^\beta
# where a_{x,y}^i is the activity of a neuron comoputed by applying kernel i at position (x,y)
# N is the total number of kernals, n is half normalization width. 
def LocalResponseNormalization(k, n, alpha, beta, name=''): 
    x = cntk.blocks.Placeholder(name='lrn_arg') 
    x2 = cntk.ops.square(x) 
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed. 
    x2s = cntk.ops.reshape(x2, (1, cntk.InferredDimension), 0, 1)
    W = cntk.ops.constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = cntk.ops.convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = cntk.ops.reshape(y, cntk.InferredDimension, 0, 2)
    den = cntk.ops.exp(beta * cntk.ops.log(k + b)) 
    apply_x = cntk.ops.element_divide(x, den)
    return cntk.blocks.Block(apply_x, 'LocalResponseNormalization', name, make_block=True)

# Create the network.
def create_alexnet():

    # Input variables denoting the features and label data
    feature_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # apply model to input
    # remove mean value 
    input = minus(feature_var, constant(114), name='mean_removed_input')
    
    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
            Convolution((11,11), 96, init=normal(0.01), pad=False, strides=(4,4), name='conv1'), 
            Activation(activation=relu, name='relu1'), 
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm1'),
            MaxPooling((3,3), (2,2), name='pool1'),
            
            Convolution((5,5), 192, init=normal(0.01), init_bias=0.1, name='conv2'), 
            Activation(activation=relu, name='relu2'), 
            LocalResponseNormalization(1.0, 2, 0.0001, 0.75, name='norm2'),
            MaxPooling((3,3), (2,2), name='pool2'),
            
            Convolution((3,3), 384, init=normal(0.01), name='conv3'), 
            Activation(activation=relu, name='relu3'), 
            Convolution((3,3), 384, init=normal(0.01), init_bias=0.1, name='conv4'), 
            Activation(activation=relu, name='relu4'), 
            Convolution((3,3), 256, init=normal(0.01), init_bias=0.1, name='conv5'), 
            Activation(activation=relu, name='relu5'), 
            MaxPooling((3,3), (2,2), name='pool5'), 
            
            Dense(4096, init=normal(0.005), init_bias=0.1, name='fc6'), 
            Activation(activation=relu, name='relu6'), 
            Dropout(0.5, name='drop6'), 
            Dense(4096, init=normal(0.005), init_bias=0.1, name='fc7'), 
            Activation(activation=relu, name='relu7'), 
            Dropout(0.5, name='drop7'),
            Dense(num_classes, init=normal(0.01), name='fc8')
            ])(input)

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits):
    # Set learning parameters
    lr_per_mb         = [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_mb, unit=cntk.learner.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = cntk.learner.momentum_schedule(0.9)
    l2_reg_weight     = 0.0005 # CNTK L2 regularization is per sample, thus same as Caffe
    
    # Create learner
    # Since we reuse parameter settings (learning rate, momentum) from Caffe, we set unit_gain to False to ensure consistency 
    parameter_learner = data_parallel_distributed_learner(
        cntk.learner.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=l2_reg_weight),
        num_quantization_bits=num_quantization_bits,
        distributed_after=0)

    # Create trainer
    return cntk.Trainer(network['output'], network['ce'], network['pe'], parameter_learner)

# Train and test
def train_and_test(network, trainer, train_source, test_source, progress_printer, minibatch_size, epoch_size):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    training_session = cntk.training_session(train_source, trainer,
        cntk.minibatch_size_schedule(minibatch_size), progress_printer, input_map, os.path.join(model_path, "AlexNet_"), epoch_size)
    training_session.train()

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    minibatch_index = 0

    while True:
        data = test_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data: break
        local_mb_samples=data[network['label']].num_samples
        metric_numer += trainer.test_minibatch(data) * local_mb_samples
        metric_denom += local_mb_samples
        minibatch_index += 1

    fin_msg = "Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom)
    progress_printer.end_progress_print(fin_msg)

    print("")
    print(fin_msg)
    print("")

    return metric_numer/metric_denom


# Train and evaluate the network.
def alexnet_train_and_eval(train_data, test_data, num_quantization_bits=32, minibatch_size=256, epoch_size = 1281167, max_epochs=112, 
                           log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False):
    _cntk_py.set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_alexnet()
    trainer = create_trainer(network, epoch_size, num_quantization_bits)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, progress_printer, minibatch_size, epoch_size)
 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', help='specify the location of your data');
    parser.add_argument('-logdir', help='specify where the training log will be saved');
    parser.add_argument('-outputdir',  help='specify where the output model/checkpoint files shall be saved');

    args = vars(parser.parse_args())

    if args['datadir'] != None:
        data_path = args['datadir']

    if args['logdir'] != None:
        log_dir = args['logdir']

    if args['outputdir'] != None:
        model_path = args['outputdir'] + "/models"

    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'val_map.txt')

    alexnet_train_and_eval(train_data, test_data, 
                           num_quantization_bits=32, 
                           max_epochs=112, 
                           log_to_file=log_dir, 
                           num_mbs_per_log=500, 
                           gen_heartbeat=True)
    Communicator.finalize()
