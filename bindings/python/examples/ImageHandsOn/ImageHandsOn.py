# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time

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

########################
# variables and stuff  #
########################

abs_path  = os.path.dirname(os.path.abspath(__file__))
cntk_path = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path = os.path.join(cntk_path, "Examples", "Image", "Datasets", "CIFAR-10")

model_dir = "./Models"

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10
features_stream_name  = 'features'
labels_stream_name    = 'labels'

def print_training_progress(trainer, mb, frequency):
    if mb % frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            mb, training_loss, eval_crit))

########################
# define the reader    #
########################

def create_reader(path, map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' does not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
                           (map_file, mean_file, cifar_py3, cifar_py3))

    image = ImageDeserializer(map_file)
    image.map_features(features_stream_name,
            [ImageDeserializer.crop(crop_type='Random', ratio=0.8,
                jitter_type='uniRatio'),
             ImageDeserializer.scale(width=image_width, height=image_height,
                 channels=num_channels, interpolations='linear'),
             ImageDeserializer.mean(mean_file)])
    image.map_labels(labels_stream_name, num_classes)

    rc = ReaderConfig(image, epoch_size=sys.maxsize)
    return rc.minibatch_source()

########################
# define the model     #
########################

def get_shape(input):
    try:
        shape = input.shape
    except AttributeError:
        output = input.output()
        shape = output.shape()
    return shape

def conv_layer(input, num_filters, filter_size, stride = (1,1), pad = 1, init = glorot_uniform(output_rank=-1, filter_rank=2), nonlinearity = relu):
    shape = get_shape(input)
    channel_count = shape[0]

    b_param = parameter(shape=(num_filters, 1, 1))
    w_param = parameter(shape=(num_filters, channel_count, filter_size[0], filter_size[1]), init=init)
    linear = convolution(w_param, input, (channel_count, stride[0], stride[1])) + b_param
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def conv_to_dense_layer(input, num_units, init=glorot_uniform(), nonlinearity = relu):
    shape = get_shape(input)

    b_param = parameter(shape=(num_units))
    w_param = parameter(shape=(shape[0], shape[1], shape[2], num_units), init=init)

    linear = b_param + times(input, w_param)
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)    
    
def dense_layer(input, num_units, init=glorot_uniform(), nonlinearity = relu):
    shape = get_shape(input)
    input_dim = shape[0]

    b_param = parameter(shape=(num_units))
    w_param = parameter(shape=(input_dim, num_units), init=init)

    linear = b_param + times(input, w_param)
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def max_pool_layer(input, pool_size, stride):
    return pooling(input, PoolingType_Max, (1, pool_size[0], pool_size[1]), (1, stride[0], stride[1]))

def create_model_1(input):
    net = {}

    net['conv1'] = conv_layer(input, 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_layer(net['pool1'], 32, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_layer(net['pool2'], 64, (5,5), init=gaussian(output_rank=-1, filter_rank=2, scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4'] = conv_to_dense_layer(net['pool3'], 64, init=gaussian(scale=12))
    net['fc5'] = dense_layer(net['fc4'], 10, init=gaussian(scale=1.5), nonlinearity = None)

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

########################
# train action         #
########################

def train(reader, max_epochs = 30):

    features_si = reader[features_stream_name]
    labels_si   = reader[labels_stream_name]

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width), features_si.m_element_type)
    label_var = input_variable((num_classes), features_si.m_element_type)

    # apply model to input
    model = create_model_1(input_var)
    z = model['fc5']

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size = 30
    minibatch_size = 64
    num_mbs_to_show_result = 100

    lr_per_sample = [0.00015625]*10+[0.000046875]*10+[0.0000156]
    momentum = 0.9**(1/minibatch_size)  # TODO: change to time constant
    l2_reg_weight = 0.03

    # trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, units=epoch_size)
    learner     = momentum_sgd(z.parameters, lr_schedule, momentum, l2_regularization_weight=l2_reg_weight)
    trainer     = Trainer(z, ce, pe, [learner])

    # process minibatches and perform model training
    mbs = 0
    original_state = reader.get_checkpoint_state()

    for epoch in range(max_epochs):
        loss_numer = 0  # TODO: find a nicer way of tracking, this is clumsy
        loss_denom = 0
        metric_numer = 0
        metric_denom = 0

        while True:
            mb = reader.next_minibatch(minibatch_size)
            if mb is None:
                reader.restore_from_checkpoint(original_state)
                break

            # Specify the mapping of input variables in the model to actual
            # minibatch data to be trained with
            data = {
                input_var: mb[features_si], 
                label_var: mb[labels_si]
                }
            trainer.train_minibatch(data)

            loss_numer += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count  # too much code for something this simple
            loss_denom +=                                           trainer.previous_minibatch_sample_count
            metric_numer += trainer.previous_minibatch_evaluation_average * trainer.previous_minibatch_sample_count
            metric_denom +=                                                 trainer.previous_minibatch_sample_count
            print_training_progress(trainer, mbs, num_mbs_to_show_result)
            mbs += 1

        print("--- EPOCH {} DONE: loss = {:0.6f} * {}, metric = {:0.1f}% * {} ---".format(epoch+1, loss_numer/loss_denom, loss_denom, metric_numer/metric_denom*100.0, metric_denom))

    return loss_numer/loss_denom, metric_numer/metric_denom

#############################
# main function boilerplate #
#############################

if __name__=='__main__':
    # TODO: get closure on Amit's feedback "Not the right pattern as we discussed over email. Please change to set_default_device(gpu(0))"
    #set_default_device(gpu(0))
    #set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works

    os.chdir(data_path)
    reader_train = create_reader(data_path, 'train_map.txt', 'CIFAR-10_mean.xml', True)
    reader_test  = create_reader(data_path, 'test_map.txt', 'CIFAR-10_mean.xml', False)

    train(reader_train)
