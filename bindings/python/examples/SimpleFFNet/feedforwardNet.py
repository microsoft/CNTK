# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgd_learner, print_training_progress, cntk_device, StreamConfiguration, text_format_minibatch_source
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error, sigmoid
from examples.common.nn import fully_connected_classifier_net

# Creates and trains a feedforward classification model
def ffnet():
    input_dim = 2
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50

    # Input variables denoting the features and label data
    input = input_variable((input_dim), np.float32)
    label = input_variable((num_output_classes), np.float32)

    # Instantiate the feedforward classification model
    netout = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    rel_path = r"../../../../Examples/Other/Simple2d/Data/SimpleDataTrain_cntk_text.txt"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    feature_stream_name = 'features'
    labels_stream_name = 'labels'


    mb_source = text_format_minibatch_source(path, list([
                    StreamConfiguration( feature_stream_name, input_dim ), 
                    StreamConfiguration( labels_stream_name, num_output_classes)]))
    features_si = mb_source.stream_info(feature_stream_name)
    labels_si = mb_source.stream_info(labels_stream_name)

    # Instantiate the trainer object to drive the model training
    lr = learning_rates_per_sample(0.02)
    trainer = Trainer(netout, ce, pe, [sgd_learner(netout.owner.parameters(), lr)])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_samples_per_sweep = 10000
    num_sweeps_to_train_with = 2
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    training_progress_output_freq = 20
    for i in range(0, int(num_minibatches_to_train)):
        mb = mb_source.get_next_minibatch(minibatch_size)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {input : mb[features_si].m_data, label : mb[labels_si].m_data}
        trainer.train_minibatch(arguments)
        print_training_progress(i, trainer, training_progress_output_freq)


if __name__=='__main__':
    # Specify the target device to be used for computing

    target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    ffnet()
