# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
import cntk
import _cntk_py

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
        features  = cntk.io.StreamDef(field='features', shape=input_dim),
        labels    = cntk.io.StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, epoch_size = cntk.io.INFINITELY_REPEAT if is_training else cntk.io.FULL_DATA_SWEEP)


# Creates and trains a feedforward classification model for MNIST images
def convnet_cifar10(debug_output=False):
    _cntk_py.set_computation_network_trace_level(0)

    image_height = 32
    image_width  = 32
    num_channels = 3
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variables denoting the features and label data
    input_var = cntk.ops.input_variable((num_channels, image_height, image_width), np.float32)
    label_var = cntk.ops.input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    input_removemean = cntk.ops.minus(input_var, cntk.ops.constant(128))
    scaled_input = cntk.ops.element_times(cntk.ops.constant(0.00390625), input_removemean)

    with cntk.layers.default_options(activation=cntk.ops.relu, pad=True): 
        z = cntk.models.Sequential([
            cntk.models.For(range(2), lambda : [
                cntk.layers.Convolution2D((3,3), 64), 
                cntk.layers.Convolution2D((3,3), 64), 
                cntk.layers.MaxPooling((3,3), (2,2))
            ]), 
            cntk.models.For(range(2), lambda i: [
                cntk.layers.Dense([256,128][i]), 
                cntk.layers.Dropout(0.5)
            ]), 
            cntk.layers.Dense(num_output_classes, activation=None)
        ])(scaled_input)
    
    ce = cntk.ops.cross_entropy_with_softmax(z, label_var)
    pe = cntk.ops.classification_error(z, label_var)

    reader_train = create_reader(os.path.join(data_path, 'Train_cntk_text.txt'), True, input_dim, num_output_classes)

    # training config
    epoch_size = 50000                  # for now we manually specify epoch size
    minibatch_size = 64

    # Set learning parameters
    lr_per_sample          = [0.0015625]*10 + [0.00046875]*10 + [0.00015625]
    lr_schedule            = cntk.learning_rate_schedule(lr_per_sample, cntk.learner.UnitType.sample, epoch_size)
    mm_time_constant       = [0]*20 + [-minibatch_size/np.log(0.9)]
    mm_schedule            = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)
    l2_reg_weight          = 0.002

    # Instantiate the trainer object to drive the model training
    learner = cntk.learner.momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                                        l2_regularization_weight = l2_reg_weight)
    trainer = cntk.Trainer(z, (ce, pe), learner)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var  : reader_train.streams.features,
        label_var  : reader_train.streams.labels
    }

    cntk.utils.log_number_of_parameters(z) ; print()
    max_epochs = 30
    progress_printer = cntk.utils.ProgressPrinter(tag='Training', num_epochs=max_epochs)

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

        progress_printer.epoch_summary(with_metric=True)
        z.save(os.path.join(model_path, "ConvNet_CIFAR10_{}.dnn".format(epoch)))
    
    # Load test data
    reader_test = create_reader(os.path.join(data_path, 'Test_cntk_text.txt'), False, input_dim, num_output_classes)

    input_map = {
        input_var  : reader_test.streams.features,
        label_var  : reader_test.streams.labels
    }

    # Test data for trained model
    epoch_size = 10000
    minibatch_size = 16

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
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    convnet_cifar10()

