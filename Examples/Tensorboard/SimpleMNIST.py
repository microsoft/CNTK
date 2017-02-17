# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, relu, element_times, constant, \
                     reduce_max, reduce_mean, reduce_min
from cntk.utils import ProgressPrinter

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from Examples.common.nn import fully_connected_classifier_net


def check_path(path):
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." %
            (path, readme_file))


def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels=StreamDef(field='labels', shape=label_dim, is_sparse=False)
    )), randomize=is_training, epoch_size=INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)


# Creates and trains a feedforward classification model for MNIST images
def simple_mnist():
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    # Input variables denoting the features and label data
    features = input_variable(input_dim, np.float32)
    label = input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant(0.00390625), features)
    netout = fully_connected_classifier_net(
        scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, relu)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    try:
        rel_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/MNIST/v0/Train-28x28_cntk_text.txt".split("/"))
    except KeyError:
        rel_path = os.path.join(*"../Image/DataSets/MNIST/Train-28x28_cntk_text.txt".split("/"))
    path = os.path.normpath(os.path.join(abs_path, rel_path))
    check_path(path)

    reader_train = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
        features: reader_train.streams.features,
        label: reader_train.streams.labels
    }

    lr_per_minibatch = learning_rate_schedule(0.2, UnitType.minibatch)
    # Instantiate the trainer object to drive the model training
    trainer = Trainer(netout, (ce, pe), sgd(netout.parameters, lr=lr_per_minibatch))

    # Instantiate a ProgressPrinter.
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_log") 
    progress_printer = ProgressPrinter(tag='Training', freq=1, tensorboard_log_dir=logdir, model=netout)

    # Get minibatches of images to train with and perform model training
    minibatch_size = 64
    num_samples_per_sweep = 6000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    for minibatch_idx in range(0, int(num_minibatches_to_train)):
        trainer.train_minibatch(reader_train.next_minibatch(minibatch_size, input_map=input_map))

        # Take snapshot of loss and eval criterion for the previous minibatch.
        progress_printer.update_with_trainer(trainer, with_metric=True)

        # Log max/min/mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        # Don't want to do that very often though, otherwise will spend too much time computing min/max/mean.
        if minibatch_idx % 10 == 9:
            for p in netout.parameters:
                progress_printer.update_value("mb_" + p.uid + "_max", reduce_max(p).eval(), minibatch_idx)
                progress_printer.update_value("mb_" + p.uid + "_min", reduce_min(p).eval(), minibatch_idx)
                progress_printer.update_value("mb_" + p.uid + "_mean", reduce_mean(p).eval(), minibatch_idx)

    # Load test data
    try:
        rel_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/MNIST/v0/Test-28x28_cntk_text.txt".split("/"))
    except KeyError:
        rel_path = os.path.join(*"../Image/DataSets/MNIST/Test-28x28_cntk_text.txt".split("/"))
    path = os.path.normpath(os.path.join(abs_path, rel_path))
    check_path(path)

    reader_test = create_reader(path, False, input_dim, num_output_classes)

    input_map = {
        features: reader_test.streams.features,
        label: reader_test.streams.labels
    }

    # Test data for trained model
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        test_result += trainer.test_minibatch(mb)

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())
    error = simple_mnist()
    print("Error: %f" % error)
