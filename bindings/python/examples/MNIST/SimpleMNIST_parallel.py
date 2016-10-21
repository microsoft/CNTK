# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, StreamConfiguration, text_format_minibatch_source, distributed
from cntk.device import cpu, set_default_device, default, DeviceDescriptor
from cntk.learner import sgd
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error, sigmoid, element_times, constant

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import fully_connected_classifier_net, print_training_progress

def check_path(path):
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

# Creates and trains a feedforward classification model for MNIST images

def simple_mnist(debug_output=False):
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    # Input variables denoting the features and label data
    input = input_variable(input_dim, np.float32)
    label = input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant(0.00390625), input)
    netout = fully_connected_classifier_net(
        scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    try:
        rel_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/MNIST/v0/Train-28x28_cntk_text.txt".split("/"))
    except KeyError:
        rel_path = os.path.join(*"../../../../Examples/Image/Datasets/MNIST/Train-28x28_cntk_text.txt".split("/"))
    path = os.path.normpath(os.path.join(abs_path, rel_path))
    check_path(path)

    feature_stream_name = 'features'
    labels_stream_name = 'labels'

    mb_source = text_format_minibatch_source(path, [
        StreamConfiguration(feature_stream_name, input_dim),
        StreamConfiguration(labels_stream_name, num_output_classes)])
    features_si = mb_source[feature_stream_name]
    labels_si = mb_source[labels_stream_name]

    # Instantiate the trainer object to drive the model training
    mpi_comm = distributed.communicator()
    workers = mpi_comm.workers()
    current_worker = mpi_comm.current_worker()
    print("List all distributed workers")
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            print("* {} {}".format(wk.global_rank, wk.host_id))
        else:
            print("  {} {}".format(wk.global_rank, wk.host_id))
            
    #mpi_comm2 = mpi_comm.sub_group([current_worker]) #feature not implemented in C++
    
    dist_trainer = distributed.trainer.data_parallel_distributed_trainer(mpi_comm, False)
    
    trainer = Trainer(netout, ce, pe, [sgd(netout.parameters,
        lr=0.003125)], distributed_trainer=dist_trainer)

    print("Training on device type:{} id:{}".format('gpu' if default().type() else 'cpu', default().id()))
        
    # Get minibatches of images to train with and perform model training
    minibatch_size = 32
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 1
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 80

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/4

    for i in range(0, int(num_minibatches_to_train)):
        mb = mb_source.next_minibatch(minibatch_size)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        arguments = {input: mb[features_si],
                     label: mb[labels_si]}
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)

    # Load test data
    try:
        rel_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/MNIST/v0/Test-28x28_cntk_text.txt".split("/"))
    except KeyError:
        rel_path = os.path.join(*"../../../../Examples/Image/Datasets/MNIST/Test-28x28_cntk_text.txt".split("/"))
    path = os.path.normpath(os.path.join(abs_path, rel_path))
    check_path(path)

    test_mb_source = text_format_minibatch_source(path, [
        StreamConfiguration(feature_stream_name, input_dim),
        StreamConfiguration(labels_stream_name, num_output_classes)], randomize=False)
    features_si = test_mb_source[feature_stream_name]
    labels_si = test_mb_source[labels_stream_name]

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = test_mb_source.next_minibatch(test_minibatch_size)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be tested with
        arguments = {input: mb[features_si],
                     label: mb[labels_si]}
        eval_error = trainer.test_minibatch(arguments)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test


if __name__=='__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())

    error = simple_mnist()
    print("Error: %f" % error)
