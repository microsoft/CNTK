# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, persist
from cntk.utils import *
from cntk.layers import *
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, relu, element_times, constant

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "MNIST")
model_path = os.path.join(abs_path, "Models")

# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='features', shape=input_dim),
        labels    = StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)


# Creates and trains a feedforward classification model for MNIST images
def convnet_mnist(debug_output=False):
    image_height = 28
    image_width  = 28
    num_channels = 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width), np.float32)
    label_var = input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant(0.00390625), input_var)
    with default_options (activation=relu, pad=False): 
        conv1 = Convolution((5,5), 32, pad=True)(scaled_input)
        pool1 = MaxPooling((3,3), (2,2))(conv1)
        conv2 = Convolution((3,3), 48)(pool1)
        pool2 = MaxPooling((3,3), (2,2))(conv2)
        conv3 = Convolution((3,3), 64)(pool2)
        f4    = Dense(96)(conv3)
        drop4 = Dropout(0.5)(f4)
        z     = Dense(num_output_classes, activation=None)(drop4)

    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    reader_train = create_reader(os.path.join(data_path, 'Train-28x28_cntk_text.txt'), True, input_dim, num_output_classes)

    # training config
    epoch_size = 60000                    # for now we manually specify epoch size
    minibatch_size = 128

    # Set learning parameters
    lr_per_sample          = [0.001]*10+[0.0005]*10+[0.0001]
    lr_schedule            = learning_rate_schedule(lr_per_sample, UnitType.sample, epoch_size)
    mm_time_constant       = [0]*5+[1024]
    mm_schedule            = momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

    # Instantiate the trainer object to drive the model training
    learner     = momentum_sgd(z.parameters, lr_schedule, mm_schedule)
    trainer     = Trainer(z, ce, pe, learner)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var  : reader_train.streams.features,
        label_var  : reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    # Get minibatches of images to train with and perform model training
    max_epochs = 40
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += data[label_var].num_samples                     # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)
        persist.save_model(z, os.path.join(model_path, "ConvNet_MNIST_{}.dnn".format(epoch)))
    
    # Load test data
    reader_test = create_reader(os.path.join(data_path, 'Test-28x28_cntk_text.txt'), False, input_dim, num_output_classes)

    input_map = {
        input_var  : reader_test.streams.features,
        label_var  : reader_test.streams.labels
    }

    # Test data for trained model
    epoch_size = 10000
    minibatch_size = 1024

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += trainer.previous_minibatch_sample_count
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    convnet_mnist()

