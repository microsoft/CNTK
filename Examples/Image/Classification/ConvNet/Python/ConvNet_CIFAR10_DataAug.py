# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import numpy as np
import cntk
import _cntk_py
import cntk.io.transforms as xforms

from cntk.layers import Convolution2D, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense, default_options, identity, Sequential, For
from cntk.layers.typing import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk import Trainer
from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk import cross_entropy_with_softmax, classification_error, relu
from cntk.ops import Function
from cntk.debugging import set_computation_network_trace_level
from cntk.logging import *

########################
# variables and paths  #
########################

# paths (are relative to current python file)
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

########################
# define the reader    #
########################

def create_reader(map_file, mean_file, is_training):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

########################
# define the model     #
########################

def create_convnet_cifar10_model(num_classes):
    with default_options(activation=relu, pad=True):
        return Sequential([
            For(range(2), lambda : [
                Convolution2D((3,3), 64), 
                Convolution2D((3,3), 64), 
                MaxPooling((3,3), strides=2)
            ]), 
            For(range(2), lambda i: [
                Dense([256,128][i]), 
                Dropout(0.5)
            ]), 
            Dense(num_classes, activation=None)
        ])

########################
# define the criteria  #
########################

# compose model function and criterion primitives into a criterion function
#  takes:   Function: features -> prediction
#  returns: Function: (features, labels) -> (loss, metric)
def create_criterion_function(model, normalize=identity):
    #@Function    # Python 3
    #def criterion(x: Tensor[(num_channels, image_height, image_width)], y: Tensor[num_classes]):
    @Function
    @Signature(x = Tensor[(num_channels, image_height, image_width)], y = Tensor[num_classes])
    def criterion(x, y):
        z = model(normalize(x))
        ce   = cross_entropy_with_softmax(z, y)
        errs = classification_error      (z, y)
        return (ce, errs)
    return criterion

########################
# train action  #
########################

def train_model(reader, reader_test, model, epoch_size=50000, max_epochs=80):

    # declare the model's input dimension
    # Training does not require this, but it is needed for deployment.
    model.update_signature((num_channels, image_height, image_width))

    # criterion function. This is what is being trained trained.
    # Model gets "sandwiched" between normalization (not part of model proper) and criterion.
    criterion = create_criterion_function(model, normalize=lambda x: x / 256)
    #debughelpers.dump_function(criterion, 'criterion')

    #from cntk.logging.graph import plot
    #plot(criterion, filename=os.path.join(model_path, "ConvNet_CIFAR10_DataAug.pdf"))

    # iteration parameters
    minibatch_size = 64
    #epoch_size = 1000 ; max_epochs = 1 # for faster testing

    # learning parameters
    learner = momentum_sgd(model.parameters, 
                           lr       = learning_rate_schedule([0.0015625]*20+[0.00046875]*20+[0.00015625]*20+[0.000046875]*10+[0.000015625], unit=UnitType.sample, epoch_size=epoch_size),
                           momentum = momentum_as_time_constant_schedule([0]*20+[600]*20+[1200], epoch_size=epoch_size),
                           l2_regularization_weight = 0.002)
    
    # trainer object
    trainer = Trainer(None, criterion, learner)

    # perform model training
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)

    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            mb = reader.next_minibatch(min(minibatch_size, epoch_size - sample_count)) # fetch minibatch.
            #trainer.train_minibatch(mb[reader.streams.features], mb[reader.streams.labels])
            trainer.train_minibatch({criterion.arguments[0]: mb[reader.streams.features], criterion.arguments[1]: mb[reader.streams.labels]})
            sample_count += mb[reader.streams.labels].num_samples                     # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
        model.save(os.path.join(model_path, "ConvNet_CIFAR10_DataAug_{}.dnn".format(epoch)))

    # return evaluation error.
    return loss, metric # return values from last epoch

########################
# eval action          #
########################

# helper function to create a dummy Trainer that one can call test_minibatch() on
# TODO: replace by a proper such class once available
def Evaluator(model, criterion):
    loss, metric = Trainer._get_loss_metric(criterion)
    parameters = set(loss.parameters)
    if model:
        parameters |= set(model.parameters)
    if metric:
        parameters |= set(metric.parameters)
    dummy_learner = momentum_sgd(tuple(parameters), 
                                 lr = learning_rate_schedule(1, UnitType.minibatch),
                                 momentum = momentum_as_time_constant_schedule(0))
    return Trainer(model, (loss, metric), dummy_learner)

def evaluate(reader, model):

    # criterion function. This is what is being evaluated
    criterion = create_criterion_function(model, normalize=lambda x: x / 256)

    # process minibatches and perform evaluation
    evaluator = Evaluator(None, criterion)

    progress_printer = ProgressPrinter(tag='Evaluation', num_epochs=1)

    while True:
        minibatch_size = 16
        mb = reader.next_minibatch(minibatch_size) # fetch minibatch
        if not mb:                                                      # until we hit the end
            break
        #metric = evaluator.test_minibatch(mb[reader.streams.features], mb[reader.streams.labels]) # evaluate minibatch
        metric = evaluator.test_minibatch({criterion.arguments[0]: mb[reader.streams.features], criterion.arguments[1]: mb[reader.streams.labels]}) # evaluate minibatch
        progress_printer.update(0, mb[reader.streams.labels].num_samples, metric) # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    # create model
    model = create_convnet_cifar10_model(num_classes=10)

    # train
    reader      = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test = create_reader(os.path.join(data_path, 'test_map.txt'),  os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    train_model(reader, reader_test, model, max_epochs=80)

    # save and load (as an illustration)
    path = data_path + "/model.cmf"
    model.save(path)

    # test
    model = Function.load(path)
    reader = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    evaluate(reader, model)
