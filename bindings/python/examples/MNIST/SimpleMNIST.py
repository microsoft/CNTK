# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, sgd_learner, create_minibatch_source, StreamConfiguration, DeviceDescriptor, text_format_minibatch_source
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error, sigmoid, element_times, constant

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import fully_connected_classifier_net, print_training_progress

# Creates and trains a feedforward classification model for MNIST images
def simple_mnist():
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    # Input variables denoting the features and label data
    input = input_variable(input_dim, np.float32)
    label = input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant((), 0.00390625), input)
    netout = fully_connected_classifier_net(scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    rel_path = os.path.join(*"../../../../Examples/Image/MNIST/Data/Train-28x28_cntk_text.txt".split("/"))
    path = os.path.normpath(os.path.join(abs_path, rel_path))
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(os.path.dirname(path), "..", "README.md"))
        raise RuntimeError("File '%s' does not exist. Please follow the instructions at %s to download and prepare it."%(path, readme_file))
    feature_stream_name = 'features'
    labels_stream_name = 'labels'
    
    mb_source = text_format_minibatch_source(path, [ 
                    StreamConfiguration( feature_stream_name, input_dim ), 
                    StreamConfiguration( labels_stream_name, num_output_classes) ])
    features_si = mb_source.stream_info(feature_stream_name)
    labels_si = mb_source.stream_info(labels_stream_name)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(netout, ce, pe, [sgd_learner(netout.owner.parameters(),
        lr=0.003125)])

    # Get minibatches of images to train with and perform model training
    minibatch_size = 32
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 1
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 20
    for i in range(0, int(num_minibatches_to_train)):
        mb = mb_source.get_next_minibatch(minibatch_size)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {input : mb[features_si].m_data, label : mb[labels_si].m_data}
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)

if __name__=='__main__':
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.gpu_device(0)
    # If it is crashing, probably you don't have a GPU, so try with CPU:
    # target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    simple_mnist()
