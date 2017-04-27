# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.device import cpu, try_set_default_device
from cntk import Trainer
from cntk.layers import Dense, Sequential, For
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.ops import input, sigmoid
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import ProgressPrinter

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

def ffnet():
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50

    # Input variables denoting the features and label data
    feature = input((input_dim), np.float32)
    label = input((num_output_classes), np.float32)

    netout = Sequential([For(range(num_hidden_layers), lambda i: Dense(hidden_layers_dim, activation=sigmoid)),
                         Dense(num_output_classes)])(feature)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    lr_per_minibatch=learning_rate_schedule(0.5, UnitType.minibatch)
    # Instantiate the trainer object to drive the model training
    learner = sgd(netout.parameters, lr=lr_per_minibatch)
    progress_printer = ProgressPrinter(128)
    trainer = Trainer(netout, (ce, pe), learner, progress_printer)

    # Get minibatches of training data and perform model training
    minibatch_size = 25

    for i in range(1024):
        features, labels = generate_random_data(
            minibatch_size, input_dim, num_output_classes)
        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        trainer.train_minibatch({feature: features, label: labels})

    trainer.summarize_training_progress()
    test_features, test_labels = generate_random_data(
        minibatch_size, input_dim, num_output_classes)
    avg_error = trainer.test_minibatch(
        {feature: test_features, label: test_labels})
    return avg_error

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())

    error = ffnet()
    print(" error rate on an unseen minibatch %f" % error)
