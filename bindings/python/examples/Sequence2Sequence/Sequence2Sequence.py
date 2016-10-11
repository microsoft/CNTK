# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import math
import time
from cntk import DeviceDescriptor, Trainer, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.learner import momentums_per_sample, momentum_sgd
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence, slice, past_value, future_value, element_select

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import LSTMP_component_with_self_stabilization, stabilize, linear_layer, print_training_progress

# Creates and trains a sequence to sequence translation model

def sequence_to_sequence_translator(debug_output=False):

    input_vocab_dim = 69
    label_vocab_dim = 69

    hidden_dim = 512
    num_layers = 2

    # Source and target inputs to the model
    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')

    input_dynamic_axes = [batch_axis, input_seq_axis]
    raw_input = input_variable(
        shape=(input_vocab_dim), dynamic_axes=input_dynamic_axes)

    label_dynamic_axes = [batch_axis, label_seq_axis]
    raw_labels = input_variable(
        shape=(label_vocab_dim), dynamic_axes=label_dynamic_axes)

    # Instantiate the sequence to sequence translation model
    input_sequence = raw_input

    # Drop the sentence start token from the label, for decoder training
    label_sequence = slice(raw_labels, label_seq_axis, 1, 0)
    label_sentence_start = sequence.first(raw_labels)

    is_first_label = sequence.is_first(label_sequence)
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)

    # Encoder
    encoder_outputH = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            encoder_outputH.output(), hidden_dim, hidden_dim, future_value, future_value)

    thought_vectorH = sequence.first(encoder_outputH)
    thought_vectorC = sequence.first(encoder_outputC)

    thought_vector_broadcastH = sequence.broadcast_as(
        thought_vectorH, label_sequence)
    thought_vector_broadcastC = sequence.broadcast_as(
        thought_vectorC, label_sequence)

    # Decoder
    decoder_history_from_ground_truth = label_sequence
    decoder_input = element_select(is_first_label, label_sentence_start_scattered, past_value(
        decoder_history_from_ground_truth))

    decoder_outputH = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0):
            recurrence_hookH = past_value
            recurrence_hookC = past_value
        else:
            isFirst = sequence.is_first(label_sequence)
            recurrence_hookH = lambda operand: element_select(
                isFirst, thought_vector_broadcastH, past_value(operand))
            recurrence_hookC = lambda operand: element_select(
                isFirst, thought_vector_broadcastC, past_value(operand))

        (decoder_outputH, encoder_outputC) = LSTMP_component_with_self_stabilization(
            decoder_outputH.output(), hidden_dim, hidden_dim, recurrence_hookH, recurrence_hookC)

    decoder_output = decoder_outputH
    decoder_dim = hidden_dim

    # Softmax output layer
    z = linear_layer(stabilize(decoder_output), label_vocab_dim)
    ce = cross_entropy_with_softmax(z, label_sequence)
    errs = classification_error(z, label_sequence)

    # Instantiate the trainer object to drive the model training
    lr = 0.007
    momentum_time_constant = 1100
    momentum_per_sample = momentums_per_sample(
        math.exp(-1.0 / momentum_time_constant))
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True

    trainer = Trainer(z, ce, errs, [momentum_sgd(z.parameters(), lr, momentum_per_sample, clipping_threshold_per_sample, gradient_clipping_with_truncation)])                   

    rel_path = r"../../../../Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train-dev-20-21.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    feature_stream_name = 'features'
    labels_stream_name = 'labels'

    mb_source = text_format_minibatch_source(path, [
        StreamConfiguration(feature_stream_name, input_vocab_dim, True, 'S0'),
        StreamConfiguration(labels_stream_name, label_vocab_dim, True, 'S1')], 10000)
    features_si = mb_source[feature_stream_name]
    labels_si = mb_source[labels_stream_name]

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 72
    training_progress_output_freq = 30
    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    while True:
        mb = mb_source.get_next_minibatch(minibatch_size)
        if len(mb) == 0:
            break

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        arguments = {raw_input: mb[features_si],
                     raw_labels: mb[labels_si]}
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)
        i += 1

    rel_path = r"../../../../Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.test.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    test_mb_source = text_format_minibatch_source(path, [
        StreamConfiguration(feature_stream_name, input_vocab_dim, True, 'S0'),
        StreamConfiguration(labels_stream_name, label_vocab_dim, True, 'S1')], 10000, False)
    features_si = test_mb_source[feature_stream_name]
    labels_si = test_mb_source[labels_stream_name]

    # choose this to be big enough for the longest sentence
    train_minibatch_size = 1024 

    # Get minibatches of sequences to test and perform testing
    i = 0
    total_error = 0.0
    while True:
        mb = test_mb_source.get_next_minibatch(train_minibatch_size)
        if len(mb) == 0:
            break

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be tested with
        arguments = {raw_input: mb[features_si],
                     raw_labels: mb[labels_si]}
        mb_error = trainer.test_minibatch(arguments)

        total_error += mb_error

        if debug_output:
            print("Minibatch {}, Error {} ".format(i, mb_error))

        i += 1

    # Average of evaluation errors of all test minibatches
    return total_error / i

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # target_device = DeviceDescriptor.cpu_device()
    # DeviceDescriptor.set_default_device(target_device)

    error = sequence_to_sequence_translator()
    print("Error: %f" % error)
