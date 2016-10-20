# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
import math
 
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff such as Linear()
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also operators like Sequential()
from cntk.utils import *
from cntk.io import ReaderConfig, ImageDeserializer
from cntk.initializer import glorot_uniform, gaussian
from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, relu
from cntk.device import gpu, set_default_device
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss

#
# variables and stuff
#
abs_path  = os.path.dirname(os.path.abspath(__file__))
cntk_path = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path = os.path.join(cntk_path, "Examples", "Image", "Datasets", "CIFAR-10")

model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10
features_stream_name  = 'features'
labels_stream_name    = 'labels'

#
# Define the reader for both training and evaluation action.
#
def create_reader(path, map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' does not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
                           (map_file, mean_file, cifar_py3, cifar_py3))

    deserializer = ImageDeserializer(map_file)
    if train:
        deserializer.map_features(features_stream_name,
            [
                ImageDeserializer.crop(crop_type='Random', ratio=0.8, jitter_type='uniRatio'),
                ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
                ImageDeserializer.mean(mean_file)
            ])
    else:
        deserializer.map_features(features_stream_name,
            [
                ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
                ImageDeserializer.mean(mean_file)
            ])

    deserializer.map_labels(labels_stream_name, num_classes)

    return ReaderConfig(deserializer, epoch_size=sys.maxsize).minibatch_source()

#
# helper APIs that define layers. 
#
def conv_layer(input, num_filters, filter_size, stride = (1,1), init = glorot_uniform(output_rank=-1, filter_rank=2), nonlinearity = relu):
    shape = input.shape
    channel_count = shape[0]

    b_param = parameter(shape=(num_filters, 1, 1))
    w_param = parameter(shape=(num_filters, channel_count, filter_size[0], filter_size[1]), init=init)
    linear = convolution(w_param, input, (channel_count, stride[0], stride[1])) + b_param
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def dense_layer(input, num_units, init=glorot_uniform(), nonlinearity = relu):

    b_param = parameter(shape=(num_units))

    if len(input.shape) >= 3:
        w_param = parameter(shape=(input.shape[0], input.shape[1], input.shape[2], num_units), init=init)
    else:
        w_param = parameter(shape=(input.shape[0], num_units), init=init)

    linear = b_param + times(input, w_param)
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def max_pool_layer(input, pool_size, stride):
    return pooling(input, PoolingType_Max, (1, pool_size[0], pool_size[1]), (1, stride[0], stride[1]))

def dropout_layer(input, p):
    return dropout(input, dropout_rate=p)

#
# define the model.
#
def create_model_1(input):
    net = {}

    net['conv1'] = conv_layer(input, 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_layer(net['pool1'], 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_layer(net['pool2'], 64, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4']   = dense_layer(net['pool3'], 64, init=gaussian(scale=12))
    net['fc5']   = dense_layer(net['fc4'], 10, init=gaussian(scale=1.5), nonlinearity = None)

    return net

def create_model_2(input):
    net = {}

    net['conv1'] = conv_layer(input, 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_layer(net['pool1'], 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_layer(net['pool2'], 64, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4']   = dense_layer(net['pool3'], 64, init=gaussian(scale=12))
    net['drop4'] = dropout_layer(net['fc4'], 0.5)
    net['fc5']   = dense_layer(net['drop4'], 10, init=gaussian(scale=1.5), nonlinearity = None)

    return net

def create_model(input):
    net = {}

    net['conv1'] = Convolution((5,5), 32, init=gaussian(output_rank=-1, filter_rank=2, scale=0.0043), activation = relu, pad = True)(input)
    net['pool1'] = Pooling(PoolingKind.MAX, (3,3), stride=(2,2))(net['conv1'])

    net['conv2'] = Convolution((5,5), 32, init=gaussian(output_rank=-1, filter_rank=2, scale=1.414), activation = relu, pad = True)(net['pool1'])
    net['pool2'] = Pooling(PoolingKind.MAX, (3,3), stride=(2,2))(net['conv2'])

    net['conv3'] = Convolution((5,5), 64, init=gaussian(output_rank=-1, filter_rank=2, scale=1.414), activation=relu, pad = True)(net['pool2'])
    net['pool3'] = Pooling(PoolingKind.MAX, (3,3), stride=(2,2))(net['conv3'])

    net['fc4'] = Dense(64, init=gaussian(scale=12), activation=relu)(net['pool3'])
    net['fc5'] = Dense(10, init=gaussian(scale=1.5), activation=relu)(net['fc4'])

    return net

#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs):

    features_slot = reader_train[features_stream_name]
    labels_slot   = reader_train[labels_stream_name]

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width), features_slot.m_element_type)
    label_var = input_variable((num_classes), labels_slot.m_element_type)

    # apply model to input
    model = create_model_1(input_var)
    z = model['fc5']

    #
    # Training action
    #

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size     = 50000
    minibatch_size = 64

    lr_per_sample = [0.00015625]*10+[0.000046875]*10+[0.0000156]
    momentum      = 0.9 ** 1/ minibatch_size
    l2_reg_weight = 0.03

    # trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, units=epoch_size)
    learner     = momentum_sgd(z.parameters, lr_schedule, momentum, 
                               l2_regularization_weight = l2_reg_weight)
    trainer     = Trainer(z, ce, pe, [learner])

    # process minibatches and perform model training
    for epoch in range(max_epochs):
        loss_numer      = 0 
        loss_denom      = 0
        metric_numer    = 0
        metric_denom    = 0
        sample_count    = 0

        while True:
            current_minibatch = min(minibatch_size, epoch_size - sample_count)
            if current_minibatch == 0:
                break

            data = reader_train.next_minibatch(current_minibatch)
            if data is None:
                break

            # minibatch data to be trained with
            trainer.train_minibatch({input_var: data[features_slot], label_var: data[labels_slot]})

            # Keep track of some statistics.
            loss_numer += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count 
            loss_denom +=                                           trainer.previous_minibatch_sample_count
            metric_numer += trainer.previous_minibatch_evaluation_average * trainer.previous_minibatch_sample_count
            metric_denom +=                                                 trainer.previous_minibatch_sample_count

            # Keep track of the number of samples processed so far.
            sample_count += data[labels_slot].num_samples
            if current_minibatch != minibatch_size:
                break

        print("Finished Epoch[{} of {}]: [Training] ce = {:0.6f} * {}, errs = {:0.1f}% * {}".format(epoch+1, max_epochs, loss_numer/loss_denom, loss_denom, metric_numer/metric_denom*100.0, metric_denom))
        trainer.save_checkpoint(os.path.join(model_path, "cifar_model"))

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

    while True:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
        if current_minibatch == 0:
            break

        data = reader_test.next_minibatch(current_minibatch)
        if data is None:
            break

        # minibatch data to be trained with
        trainer.test_minibatch({input_var: data[features_slot], label_var: data[labels_slot]})

        # Keep track of some statistics.
        metric_numer += trainer.previous_minibatch_evaluation_average * trainer.previous_minibatch_sample_count
        metric_denom +=                                                 trainer.previous_minibatch_sample_count

        # Keep track of the number of samples processed so far.
        sample_count += data[labels_slot].num_samples
        minibatch_index += 1
        if current_minibatch != minibatch_size:
            break

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

if __name__=='__main__':
    os.chdir(data_path)
    reader_train = create_reader(data_path, 'train_map.txt', 'CIFAR-10_mean.xml', True)
    reader_test  = create_reader(data_path, 'test_map.txt', 'CIFAR-10_mean.xml', False)

    train_and_evaluate(reader_train, reader_test, 10)
