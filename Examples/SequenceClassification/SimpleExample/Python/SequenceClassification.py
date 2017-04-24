# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
from cntk import Trainer, Axis #, text_format_minibatch_source, StreamConfiguration
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.device import cpu, try_set_default_device
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.ops import input, sequence
from cntk.logging import ProgressPrinter
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error

from cntk.layers import Sequential, Embedding, Recurrence, LSTM, Dense

# Creates the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifier_net(feature, num_output_classes, embedding_dim, LSTM_dim, cell_dim):
    lstm_classifier = Sequential([Embedding(embedding_dim),
                                  Recurrence(LSTM(LSTM_dim, cell_dim))[0],
                                  sequence.last,
                                  Dense(num_output_classes)])
    return lstm_classifier(feature)

# Creates and trains a LSTM sequence classification model
def train_sequence_classifier():
    input_dim = 2000
    cell_dim = 25
    hidden_dim = 25
    embedding_dim = 50
    num_output_classes = 5

    # Input variables denoting the features and label data
    features = sequence.input(shape=input_dim, is_sparse=True)
    label = input(num_output_classes)

    # Instantiate the sequence classification model
    classifier_output = LSTM_sequence_classifier_net(
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
    progress_printer = ProgressPrinter(10)
    trainer = Trainer(classifier_output, (ce, pe),
                      sgd(classifier_output.parameters, lr=lr_per_sample),
                      progress_printer)

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200

    for i in range(251):
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)

    import copy

    evaluation_average = copy.copy(
        trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)

    return evaluation_average, loss_average

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())

    error, _ = train_sequence_classifier()
    print("Error: %f" % error)
