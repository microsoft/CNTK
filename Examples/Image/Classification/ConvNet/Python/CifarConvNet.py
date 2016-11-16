# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import numpy as np

from cntk.blocks import default_options, Placeholder, identity
from cntk.layers import Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense
from cntk.models import Sequential, For
from cntk.utils import *
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.initializer import glorot_uniform
from cntk import Trainer, Evaluator
from cntk.learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error, relu
from cntk.ops import input_variable, constant, parameter, element_times, combine
from cntk.persist import load_model, save_model

########################
# variables and paths  #
########################

# paths (are relative to current python file)
abs_path   = os.path.dirname(os.path.abspath(__file__))
cntk_path  = os.path.normpath(os.path.join(abs_path, "..", "..", "..", "..", ".."))
data_path  = os.path.join(cntk_path, "Examples", "Image", "DataSets", "CIFAR-10")
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
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from Examples/Image/DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
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
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

########################
# define the model     #
########################

#
# VGG like network for Cifar dataset.
#
#       | VGG9          |
#       | ------------- |
#       | conv3-64      |
#       | conv3-64      |
#       | max3          |
#       |               |
#       | conv3-96      |
#       | conv3-96      |
#       | max3          |
#       |               |
#       | conv3-128     |
#       | conv3-128     |
#       | max3          |
#       |               |
#       | FC-1024       |
#       | FC-1024       |
#       |               |
#       | FC-10         |
#
def create_vgg9_model(num_classes):
    with default_options(activation=relu):
        return Sequential([
            For(range(3), lambda i: [
                Convolution((3,3), [64,96,128][i], init=glorot_uniform(), pad=True),
                Convolution((3,3), [64,96,128][i], init=glorot_uniform(), pad=True),
                MaxPooling((3,3), strides=(2,2))
            ]),
            For(range(2), lambda:
                Dense(1024, init=glorot_uniform())
            ),
            Dense(num_classes, init=glorot_uniform(), activation=None)
        ])

########################
# define the criteria  #
########################

_1, _2 = (Placeholder(), Placeholder())
_ = Placeholder()

# compose model function (with optional input normalization) and criterion primitives into a criterion function
#  takes:   Function: features -> prediction
#  returns: Function: (features, labels) -> (loss, metric)
# This function is generic and could be a stock function create_ce_classification_criterion().
def create_criterion_function(model, normalize=identity):
    z = model(normalize(_1))
    ce   = cross_entropy_with_softmax(z, _2, name='_ce')
    errs = classification_error      (z, _2, name='_errs')
    return combine ([ce, errs]) # (features, labels) -> (loss, metric)

########################
# train & eval action  #
########################

def train_and_evaluate(reader, reader_test, model, max_epochs):

    # declare the model's input dimension
    model.replace_placeholders({model.placeholders[0]: input_variable((num_channels, image_height, image_width))})
    # BUGBUG: ^^ Trainer requires this, although the criterion roots are not part of this.

    # criterion function. This is what is being trained trained.
    # Model gets "sandwiched" between normalization (not part of model proper) and criterion.
    criterion = create_criterion_function(model, normalize=element_times(1.0 / 256.0, _))
    criterion.replace_placeholders({criterion.placeholders[0]: input_variable((num_channels, image_height, image_width)), criterion.placeholders[1]: input_variable((num_classes))})

    # iteration parameters
    epoch_size     = 50000
    minibatch_size = 64
    #epoch_size = 1000 ; max_epochs = 1 # for faster testing

    # learning parameters
    learner = momentum_sgd(model.parameters, 
                           lr = learning_rate_schedule([0.1]*10 + [0.03]*10 + [0.01], UnitType.minibatch, epoch_size),
                           momentum = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9)),
                           l2_regularization_weight = 0.0001)

    # trainer object
    trainer = Trainer(model, criterion.outputs[0], criterion.outputs[1], learner)

    # perform model training
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            mb = reader.next_minibatch(min(minibatch_size, epoch_size - sample_count)) # fetch minibatch.
            #trainer.train_minibatch({ trainer.loss_function.arguments[0]: mb[reader.streams.features], trainer.loss_function.arguments[1]: mb[reader.streams.labels] }) # update model with it
            # BUGBUG: ^^ Fails with "Function::Forward: Required argument's () value that the requested output(s) depend on has not been provided"
            trainer.train_minibatch({ criterion.arguments[0]: mb[reader.streams.features], criterion.arguments[1]: mb[reader.streams.labels] }) # update model with it
            # TODO: We should just be able to say train_minibatch(mb[reader.streams.features], mb[reader.streams.labels])
            sample_count += mb[reader.streams.labels].num_samples                     # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    #return metric_numer/metric_denom


    # evaluate with current Trainer instance; just to make sure we save and load the model correctly and BN works now --TODO: delete once confirmed
    epoch_size     = 10000
    minibatch_size = 16
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        mbsize = min(minibatch_size, epoch_size - sample_count)
        mb = reader_test.next_minibatch(mbsize)
        metric_numer += mbsize * trainer.test_minibatch({ criterion.arguments[0]: mb[reader_test.streams.features], criterion.arguments[1]: mb[reader_test.streams.labels] })
        metric_denom += mbsize
        sample_count += mb[reader_test.streams.labels].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    # return evaluation error.
    return loss, metric # return values from last epoch

########################
# eval action          #
########################

def evaluate(reader, model):

    # criterion function. This is what is being evaluated
    criterion = create_criterion_function(model, normalize=element_times(1.0 / 256.0, _))
    criterion.replace_placeholders({criterion.placeholders[0]: input_variable((num_channels, image_height, image_width)), criterion.placeholders[1]: input_variable((num_classes))})

    # process minibatches and perform evaluation
    evaluator = Evaluator(model, criterion.outputs[0], criterion.outputs[1])

    progress_printer = ProgressPrinter(tag='Evaluation')

    while True:
        minibatch_size = 1000
        mb = reader.next_minibatch(minibatch_size) # fetch minibatch
        if not mb:                                                      # until we hit the end
            break
        metric = evaluator.test_minibatch({ criterion.arguments[0]: mb[reader.streams.features], criterion.arguments[1]: mb[reader.streams.labels] }) # evaluate minibatch
        progress_printer.update(0, mb[reader.streams.labels].num_samples, metric) # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    # create model
    model = create_vgg9_model(10)

    # train
    reader = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'),  os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    train_and_evaluate(reader, reader_test, model, max_epochs=5)
    #
    ## save and load (as an illustration)
    path = data_path + "/model.cmf"
    save_model(model, path)
    model1 = load_model(path)

    # test
    reader = create_reader(os.path.join(data_path, 'test_map.txt'),  os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    evaluate(reader, model1)
