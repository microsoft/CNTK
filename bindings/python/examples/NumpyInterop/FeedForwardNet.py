# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import DeviceDescriptor, Trainer, cntk_device, StreamConfiguration
from cntk.learner import sgd
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error, sigmoid

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import fully_connected_classifier_net, print_training_progress

# make sure we get always the same "randomness"
np.random.seed(0)

def generate_random_data(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

# Creates and trains a feedforward classification model

def ffnet(debug_output=False):
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50

    # Input variables denoting the features and label data
    input = input_variable((input_dim), np.float32)
    label = input_variable((num_output_classes), np.float32)

    # Instantiate the feedforward classification model
    netout = fully_connected_classifier_net(
        input, num_output_classes, hidden_layers_dim, num_hidden_layers, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(netout, ce, pe, [sgd(netout.parameters(), lr=0.02)])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_samples_per_sweep = 10000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (
        num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 60

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    for i in range(0, int(num_minibatches_to_train)):
        features, labels = generate_random_data(
            minibatch_size, input_dim, num_output_classes)
        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        trainer.train_minibatch({input: features, label: labels})
        print_training_progress(trainer, i, training_progress_output_freq)

    test_features, test_labels = generate_random_data(
        minibatch_size, input_dim, num_output_classes)
    avg_error = trainer.test_minibatch(
        {input: test_features, label: test_labels})
    return avg_error

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # target_device = DeviceDescriptor.cpu_device()
    # DeviceDescriptor.set_default_device(target_device)

    error = ffnet()
    print("Error: %f" % error)
