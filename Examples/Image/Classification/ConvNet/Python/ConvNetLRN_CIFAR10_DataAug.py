# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import numpy as np
import cntk as C
import _cntk_py
import cntk.io.transforms as xforms

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file, is_training):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='RandomSide', side_ratio=0.8, jitter_type='uniRatio') # train uses jitter
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features=C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels=C.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)

# Local Response Normalization layer. See Section 3.3 of the paper:
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# The mathematical equation is:
#   b_{x,y}^i=a_{x,y}^i/(k+\alpha\sum_{j=max(0,i-n)}^{min(N-1, i+n)}(a_{x,y}^j)^2)^\beta
# where a_{x,y}^i is the activity of a neuron comoputed by applying kernel i at position (x,y)
# N is the total number of kernals, n is half normalization width.
def LocalResponseNormalization(k, n, alpha, beta, name=''):
    x = C.placeholder(name='lrn_arg')
    x2 = C.square(x)
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
    x2s = C.reshape(x2, (1, C.InferredDimension), 0, 1)
    W = C.constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = C.convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = C.reshape(y, C.InferredDimension, 0, 2)
    den = C.exp(beta * C.log(k + b))
    apply_x = C.element_divide(x, den)
    return apply_x

# Train and evaluate the network.
def convnetlrn_cifar10_dataaug(reader_train, reader_test, epoch_size=50000, max_epochs = 80):
    _cntk_py.set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # apply model to input
    scaled_input = C.element_times(C.constant(0.00390625), input_var)

    with C.layers.default_options (activation=C.relu, pad=True):
        z = C.layers.Sequential([
            C.layers.For(range(2), lambda : [
                C.layers.Convolution2D((3,3), 64),
                C.layers.Convolution2D((3,3), 64),
                LocalResponseNormalization (1.0, 4, 0.001, 0.75),
                C.layers.MaxPooling((3,3), (2,2))
            ]),
            C.layers.For(range(2), lambda i: [
                C.layers.Dense([256,128][i]),
                C.layers.Dropout(0.5)
            ]),
            C.layers.Dense(num_classes, activation=None)
        ])(scaled_input)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    # training config
    minibatch_size = 64

    # Set learning parameters
    lr_per_sample          = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule            = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=epoch_size)
    mms                    = [0]*20 + [0.9983347214509387]*20 + [0.9991670137924583]
    mm_schedule            = C.learners.momentum_schedule_per_sample(mms, epoch_size=epoch_size)
    l2_reg_weight          = 0.002

    # trainer object
    learner = C.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                                        unit_gain = True,
                                        l2_regularization_weight = l2_reg_weight)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z) ; print()

    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far

        trainer.summarize_training_progress()
        z.save(os.path.join(model_path, "ConvNet_CIFAR10_DataAug_{}.dnn".format(epoch)))

    ### Evaluation action
    epoch_size     = 10000
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
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)

    convnetlrn_cifar10_dataaug(reader_train, reader_test)

