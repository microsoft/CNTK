﻿# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgd_learner, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import LSTMP_component_with_self_stabilization, embedding, linear_layer, select_last, print_training_progress

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(input, num_output_classes, embedding_dim, LSTM_dim, cell_dim):
    embedding_function = embedding(input, embedding_dim)
    LSTM_function = LSTMP_component_with_self_stabilization(embedding_function, LSTM_dim, cell_dim)[0]
    thought_vector = select_last(LSTM_function)

    return linear_layer(thought_vector, num_output_classes)

# Creates and trains a LSTM sequence classification model
def train_sequence_classifier():
    input_dim = 2000;
    cell_dim = 25;
    hidden_dim = 25;
    embedding_dim = 50;
    num_output_classes = 5;

    # Input variables denoting the features and label data
    features = input_variable(shape=input_dim, is_sparse=True)
    label = input_variable(num_output_classes, dynamic_axes = [Axis.default_batch_axis()])

    # Instantiate the sequence classification model
    classifier_output = LSTM_sequence_classifer_net(features, num_output_classes, embedding_dim, hidden_dim, cell_dim)

    ce = cross_entropy_with_softmax(classifier_output, label)
    pe = classification_error(classifier_output, label)

    rel_path = r"../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    feature_stream_name = 'features'
    labels_stream_name = 'labels'

    mb_source = text_format_minibatch_source(path, [
                    StreamConfiguration( feature_stream_name, input_dim, True, 'x' ),
                    StreamConfiguration( labels_stream_name, num_output_classes, False, 'y')], 0)

    features_si = mb_source.stream_info(features)
    labels_si = mb_source.stream_info(label)

    # Instantiate the trainer object to drive the model training
    lr = lr = learning_rates_per_sample(0.0005)
    trainer = Trainer(classifier_output, ce, pe, [sgd_learner(classifier_output.owner.parameters(), lr)])

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200
    training_progress_output_freq = 10
    i = 0;
    while True:
        mb = mb_source.get_next_minibatch(minibatch_size)
        if  len(mb) == 0:
            break

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {features : mb[features_si].m_data, label : mb[labels_si].m_data}
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)

        i += 1

if __name__=='__main__':
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    train_sequence_classifier()
