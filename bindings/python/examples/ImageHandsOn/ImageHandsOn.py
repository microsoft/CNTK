# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import math
 
from cntk.blocks import *  # non-layer like building blocks such as LSTM()
from cntk.layers import *  # layer-like stuff
from cntk.models import *  # higher abstraction level, e.g. entire standard models and also operators like Sequential()
from cntk.utils import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.initializer import glorot_uniform, he_normal
from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, relu, convolution, pooling, PoolingType_Max

#
# Paths relative to current python file.
#
abs_path   = os.path.dirname(os.path.abspath(__file__))
cntk_path  = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path  = os.path.join(cntk_path, "Examples", "Image", "Datasets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10

#
# Define the reader for both training and evaluation action.
#
def create_reader(path, map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' does not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
                           (map_file, mean_file, cifar_py3, cifar_py3))

    # transformation pipeline for the features differs between training and test
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
    # ImageDeserializer exposes two fields with fixed names 'image' (first column of map file) and 'label' (second column)
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms),
        labels   = StreamDef(field='label', shape=num_classes)
    )))
    #if train:
    #    deserializer.map_features(features_stream_name,
    #        [
    #            ImageDeserializer.crop(crop_type='Random', ratio=0.8, jitter_type='uniRatio'),
    #            ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    #            ImageDeserializer.mean(mean_file)
    #        ])
    #else:
    #    deserializer.map_features(features_stream_name,
    #        [
    #            ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    #            ImageDeserializer.mean(mean_file)
    #        ])
    #
    #deserializer.map_labels(labels_stream_name, num_classes)

    #return ReaderConfig(deserializer, epoch_size = sys.maxsize).minibatch_source()
    #return MinibatchSource(deserializer)

#
# helper APIs that define layers and shows low level APIs usage. 
#
def conv_layer(input, num_filters, filter_size, init, strides=(1,1), nonlinearity=relu):
    if nonlinearity is None:
        nonlinearity = lambda x: x

    channel_count = input.shape[0]

    b_param = parameter(shape=(num_filters, 1, 1))
    w_param = parameter(shape=(num_filters, channel_count, filter_size[0], filter_size[1]), init=init)
    r       = convolution(w_param, input, (channel_count, strides[0], strides[1])) + b_param
    r       = nonlinearity(r)

    return r

def bn_layer(input, spatial_rank, nonlinearity=relu, bn_time_const=5000, b_value=0, sc_value=1):
    if nonlinearity is None:
        nonlinearity = lambda x: x

    dims = input.shape[0]

    bias_params    = parameter((dims), init=b_value)
    scale_params   = parameter((dims), init=sc_value)
    running_mean   = constant(0, (dims))
    running_invstd = constant(0, (dims))

    r = batch_normalization(input, scale_params, bias_params, running_mean, running_invstd, spatial_rank > 0, bn_time_const, use_cudnn_engine=True)
    r = nonlinearity(r)
    return r

def conv_bn_layer(input, num_filters, filter_size, init, strides=(1,1), nonlinearity=relu, bn_time_const=5000, b_value=0, sc_value=1):
    r = conv_layer(input, num_filters, filter_size, init, strides=strides, nonlinearity=None)
    r = bn_layer(r, 2, nonlinearity=nonlinearity)
    return r

def dense_layer(input, num_units, init, nonlinearity=relu):
    if nonlinearity is None:
        nonlinearity = lambda x: x

    b_param = parameter(shape=(num_units))

    if len(input.shape) >= 3:
        w_param = parameter(shape=(input.shape[0], input.shape[1], input.shape[2], num_units), init=init)
    else:
        w_param = parameter(shape=(input.shape[0], num_units), init=init)

    r = b_param + times(input, w_param)
    r = nonlinearity(r)
    return r

def dense_bn_layer(input, num_units, init, nonlinearity=relu):
    r = dense_layer(input, num_units, init, nonlinearity=None)
    r = bn_layer(r, 0, nonlinearity=nonlinearity)
    return r

def max_pool_layer(input, pool_size, stride):
    return pooling(input, PoolingType_Max, (1, pool_size[0], pool_size[1]), (1, stride[0], stride[1]))

def dropout_layer(input, rate):
    return dropout(input, dropout_rate=rate)

# HACK: express the outdated gaussian() initializer as a he_normal
# TODO: replace all gaussian() calls by inlining
#from cntk.initializer import gaussian
def gaussian(scale=1):
    return he_normal(scale=scale * math.sqrt(0.02))

# Define basic model
def create_basic_model(input):
    net = {}

    net['conv1'] = conv_layer(input, 32, (5,5), init = gaussian(scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_layer(net['pool1'], 32, (5,5), init = gaussian(scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_layer(net['pool2'], 64, (5,5), init = gaussian(scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4']   = dense_layer(net['pool3'], 64, init = gaussian(scale=12))
    net['fc5']   = dense_layer(net['fc4'], 10, init = gaussian(scale=1.5), nonlinearity = None)

    return net

# Task 1: Adding Dropout
def create_basic_model_with_dropout(input):
    net = {}
    net['conv1'] = conv_layer(input, 32, (5,5), init = gaussian(scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_layer(net['pool1'], 32, (5,5), init = gaussian(scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_layer(net['pool2'], 64, (5,5), init = gaussian(scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4']   = dense_layer(net['pool3'], 64, init = gaussian(scale=12))
    net['drop4'] = dropout_layer(net['fc4'], 0.75)
    net['fc5']   = dense_layer(net['drop4'], 10, init = gaussian(scale=1.5), nonlinearity=None)

    return net

# Task 2: Adding Batch Normalization
def create_basic_model_with_batch_normalization(input):
    net = {}
    net['conv1'] = conv_bn_layer(input, 32, (5,5), init = gaussian(scale=0.0043))
    net['pool1'] = max_pool_layer(net['conv1'], (3,3), (2,2))

    net['conv2'] = conv_bn_layer(net['pool1'], 32, (5,5), init = gaussian(scale=1.414))
    net['pool2'] = max_pool_layer(net['conv2'], (3,3), (2,2))

    net['conv3'] = conv_bn_layer(net['pool2'], 64, (5,5), init = gaussian(scale=1.414))
    net['pool3'] = max_pool_layer(net['conv3'], (3,3), (2,2))

    net['fc4']   = dense_bn_layer(net['pool3'], 64, init = gaussian(scale=12))
    net['fc5']   = dense_layer(net['fc4'], 10, init = gaussian(scale=1.5), nonlinearity=None)

    return net

# Task 3: Implement Task 1 and 2 using layer API
def create_basic_model_layer(input):
    net = {}

    with default_options(activation=relu):
        #with default_options_for(Convolution, pad=True):  # TODO: implement this
            model = Sequential([
                LayerStack(3, lambda i:
                [
                    Convolution((5,5), [32,32,64][i], init=gaussian(scale=[0.0043,1.414,1.414][i]), pad=True),
                    MaxPooling((3,3), strides=(2,2))    #, BatchNormalization(spatial_rank=2)  # Note: I know this BN is not the same as above
                ]),
                Dense(64, init=gaussian(scale=12)),
                Dense(10, init=gaussian(scale=1.5), activation=None)
            ])

    # TODO: unify the patterns
    net['fc5'] = model(input)

    return net

#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs):

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # apply model to input
    model = create_basic_model(input_var)
    #model = create_basic_model_layer(input_var)
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

    # For basic model
    lr_per_sample       = [0.00015625]*10+[0.000046875]*10+[0.0000156]
    momentum_per_sample = 0.9 ** (1.0 / minibatch_size)  # BUGBUG: why does this work? Should be as time const, no?
    l2_reg_weight       = 0.03

    # For basic model with batch normalization
    # lr_per_sample       = [0.00046875]*7+[0.00015625]*10+[0.000046875]*10+[0.000015625]
    # momentum_per_sample = 0
    # l2_reg_weight       = 0

    # trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, units=epoch_size)
    learner     = momentum_sgd(z.parameters, lr_schedule, momentum_per_sample, 
                               l2_regularization_weight = l2_reg_weight)
    trainer     = Trainer(z, ce, pe, [learner])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    # process minibatches and perform model training
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            current_minibatch = min(minibatch_size, epoch_size - sample_count)

            # fetch next mini batch.
            data = reader_train.next_minibatch(current_minibatch, input_map=input_map)

            # minibatch data to be trained with
            trainer.train_minibatch(data)

            # keep track of the number of samples processed so far
            sample_count += data[label_var].num_samples

            # progress
            progress_printer.update_with_trainer(trainer, with_metric=True)

        progress_printer.epoch_summary(with_metric=True)
    
    #
    # Evaluation action
    # TODO: This should be a separate function call.
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
        data = reader_train.next_minibatch(current_minibatch, input_map=input_map)
        #data = reader_test.next_minibatch(current_minibatch)

        # minibatch data to be trained with
        #metric_numer += trainer.test_minibatch(input_map) * current_minibatch
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1
        if current_minibatch != minibatch_size:
            break

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

if __name__=='__main__':
    os.chdir(data_path) # BUGBUG: This is only needed because ImageReader uses relative paths in the map file. Ugh.

    # TODO: leave these in for now as debugging aids; remove for beta
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    #set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    reader_train = create_reader(data_path, 'train_map.txt', 'CIFAR-10_mean.xml', True)
    reader_test  = create_reader(data_path, 'test_map.txt',  'CIFAR-10_mean.xml', False)

    # temp for Frank, to be removed
    #data_path = abs_path
    #os.chdir(data_path)
    #reader_train = create_reader(data_path, 'cifar-10-batches-py/train_map.txt', 'cifar-10-batches-py/CIFAR-10_mean.xml', True)
    #reader_test  = create_reader(data_path, 'cifar-10-batches-py/test_map.txt',  'cifar-10-batches-py/CIFAR-10_mean.xml', False)

    train_and_evaluate(reader_train, reader_test, max_epochs=10)
