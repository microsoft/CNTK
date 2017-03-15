# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import argparse
import numpy as np
import sys
import os
from cntk import Trainer, minibatch_size_schedule 
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, relu, element_times, constant
from cntk.training_session import *
from cntk.utils import ProgressPrinter, TensorBoardProgressWriter

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "common"))
from nn import fully_connected_classifier_net

def check_path(path):
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels    = StreamDef(field='labels',   shape=label_dim, is_sparse=False)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)


# Creates and trains a feedforward classification model for MNIST images

def simple_mnist(tensorboard_logdir=None):
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    # Input variables denoting the features and label data
    input = input_variable(input_dim, np.float32)
    label = input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant(0.00390625), input)
    z = fully_connected_classifier_net(
        scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, relu)

    ce = cross_entropy_with_softmax(z, label)
    pe = classification_error(z, label)

    data_dir = os.path.join(abs_path, "..", "..", "..", "DataSets", "MNIST")

    path = os.path.normpath(os.path.join(data_dir, "Train-28x28_cntk_text.txt"))
    check_path(path)

    reader_train = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
        input  : reader_train.streams.features,
        label  : reader_train.streams.labels
    }

    # Training config
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10

    # Instantiate progress writers.
    #training_progress_output_freq = 100
    progress_writers = [ProgressPrinter(
        #freq=training_progress_output_freq,
        tag='Training',
        num_epochs=num_sweeps_to_train_with)]

    if tensorboard_logdir is not None:
        progress_writers.append(TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z))

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.2, UnitType.minibatch)
    trainer = Trainer(z, (ce, pe), sgd(z.parameters, lr=lr_per_minibatch), progress_writers)

    training_session(
        trainer=trainer,
        mb_source = reader_train,
        mb_size = minibatch_size,
        var_to_stream = input_map,
        max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
        progress_frequency=num_samples_per_sweep
    ).train()
    
    # Load test data
    path = os.path.normpath(os.path.join(data_dir, "Test-28x28_cntk_text.txt"))
    check_path(path)

    reader_test = create_reader(path, False, input_dim, num_output_classes)

    input_map = {
        input  : reader_test.streams.features,
        label  : reader_test.streams.labels
    }

    # Test data for trained model
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test


if __name__=='__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())

    parser = argparse.ArgumentParser()
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)
    args = vars(parser.parse_args())

    error = simple_mnist(args['tensorboard_logdir'])
    print("Error: %f" % error)
