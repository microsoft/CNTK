# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
from cntk import Trainer, Axis #, text_format_minibatch_source, StreamConfiguration
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.device import cpu, set_default_device
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sequence

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "common"))
from nn import LSTMP_component_with_self_stabilization, embedding, linear_layer, print_training_progress

# Creates the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(input, num_output_classes, embedding_dim, LSTM_dim, cell_dim):
    embedding_function = embedding(input, embedding_dim)
    LSTM_function = LSTMP_component_with_self_stabilization(
        embedding_function.output, LSTM_dim, cell_dim)[0]
    thought_vector = sequence.last(LSTM_function)

    return linear_layer(thought_vector, num_output_classes)

# Creates and trains a LSTM sequence classification model
def train_sequence_classifier(debug_output=False):
    input_dim = 2000
    cell_dim = 25
    hidden_dim = 25
    embedding_dim = 50
    num_output_classes = 5

    # Input variables denoting the features and label data
    features = input_variable(shape=input_dim, is_sparse=True)
    label = input_variable(num_output_classes, dynamic_axes=[
                           Axis.default_batch_axis()])

    # Instantiate the sequence classification model
    classifier_output = LSTM_sequence_classifer_net(
        features, num_output_classes, embedding_dim, hidden_dim, cell_dim)

    ce = cross_entropy_with_softmax(classifier_output, label)
    pe = classification_error(classifier_output, label)

    rel_path = r"../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    reader = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
        features : reader.streams.features,
        label    : reader.streams.labels
    }

    lr_per_sample = learning_rate_schedule(0.0005, UnitType.sample)
    # Instantiate the trainer object to drive the model training
    trainer = Trainer(classifier_output, (ce, pe),
                      sgd(classifier_output.parameters, lr=lr_per_sample))

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200
    training_progress_output_freq = 10

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    for i in range(251):
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
        print_training_progress(trainer, i, training_progress_output_freq)

    import copy

    evaluation_average = copy.copy(
        trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)

    return evaluation_average, loss_average

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())

    error, _ = train_sequence_classifier()
    print("Error: %f" % error)
