# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import numpy as np

from cntk.utils import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.initializer import glorot_uniform
from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, relu, convolution, pooling, AVG_POOLING
from cntk.ops import input_variable, constant, parameter, combine, times, element_times

#
# Paths relative to current python file.
#
abs_path   = os.path.dirname(os.path.abspath(__file__))
cntk_path  = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path  = os.path.join(cntk_path, "Examples", "Image", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

#
# Layer wrappers
#
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import conv_bn_relu_layer, conv_bn_layer, linear_layer

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train, distributed_communicator=None):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from Examples/Image/DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            ImageDeserializer.crop(crop_type='Random', ratio=0.8, jitter_type='uniRatio') # train uses jitter
        ]
    transforms += [
        ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        ImageDeserializer.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes))),      # and second as 'label'
        distributed_communicator=distributed_communicator)

#
# Resnet building blocks
#
def resnet_basic(input, out_feature_map_count, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    p = c2 + input
    return relu(p)

def resnet_basic_inc(input, out_feature_map_count, strides, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, [3, 3], strides, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    s  = conv_bn_layer(input, out_feature_map_count, [1, 1], strides, bn_time_const)
    p = c2 + s
    return relu(p)

def resnet_basic_stack2(input, out_feature_map_count, bn_time_const):
    r1 = resnet_basic(input, out_feature_map_count, bn_time_const)
    r2 = resnet_basic(r1, out_feature_map_count, bn_time_const)
    return r2

def resnet_basic_stack3(input, out_feature_map_count, bn_time_const):
    r12 = resnet_basic_stack2(input, out_feature_map_count, bn_time_const)
    r3 = resnet_basic(r12, out_feature_map_count, bn_time_const)
    return r3

#   
# Defines the residual network model for classifying images
#
def create_resnet_model(input, num_classes):
    bn_time_const = 4096

    c_map1 = 16
    
    feat_scale = 0.00390625

    input_norm = element_times(feat_scale, input)

    conv = conv_bn_relu_layer(input_norm, c_map1, [3, 3], [1, 1], bn_time_const)
    r1_1 = resnet_basic_stack3(conv, c_map1, bn_time_const)

    c_map2 = 32

    r2_1 = resnet_basic_inc(r1_1, c_map2, [2, 2], bn_time_const)
    r2_2 = resnet_basic_stack2(r2_1, c_map2, bn_time_const)

    c_map3 = 64
    r3_1 = resnet_basic_inc(r2_2, c_map3, [2, 2], bn_time_const)
    r3_2 = resnet_basic_stack2(r3_1, c_map3, bn_time_const)

    # Global average pooling
    poolw = 8
    poolh = 8
    poolh_stride = 1
    poolv_stride = 1

    pool = pooling(r3_2, AVG_POOLING, (1, poolh, poolw), (1, poolv_stride, poolh_stride))
    return linear_layer(pool, num_classes)

#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs):

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # apply model to input
    z = create_resnet_model(input_var, 10)

    #
    # Training action
    #

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size     = 50000
    minibatch_size = 128

    # Set learning parameters
    lr_per_minibatch       = learning_rate_schedule([1]*80 + [0.1]*40 + [0.01], epoch_size, UnitType.minibatch)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.0001
    
    # trainer object
    learner     = momentum_sgd(z.parameters, lr = lr_per_minibatch, 
                               momentum = momentum_time_constant,
                               l2_regularization_weight = l2_reg_weight)
    trainer     = Trainer(z, ce, pe, learner)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)
    
    #
    # Evaluation action
    #
    epoch_size     = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Eval')
    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)

    train_and_evaluate(reader_train, reader_test, max_epochs=5)
