import sys
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs,\
        INFINITELY_REPEAT
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk import input, cross_entropy_with_softmax, \
        classification_error, sequence
from cntk.logging import ProgressPrinter
from cntk.layers import Sequential, Embedding, Recurrence, LSTM, Dense

# Creates the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='x', shape=input_dim, is_sparse=True),
        labels=StreamDef(field='y', shape=label_dim, is_sparse=False)
        )), randomize=is_training,
        max_sweeps=INFINITELY_REPEAT if is_training else 1)


# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifier_net(input, num_output_classes, embedding_dim,
                                LSTM_dim, cell_dim):
    lstm_classifier = Sequential([Embedding(embedding_dim),
                                  Recurrence(LSTM(LSTM_dim, cell_dim))[0],
                                  sequence.last,
                                  Dense(num_output_classes)])
    return lstm_classifier(input)


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

    rel_path = ("../../../Tests/EndToEndTests/Text/" +
                "SequenceClassification/Data/Train.ctf")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    reader = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
            features: reader.streams.features,
            label:    reader.streams.labels
    }

    lr_per_sample = learning_rate_schedule(0.0005, UnitType.sample)
    # Instantiate the trainer object to drive the model training
    progress_printer = ProgressPrinter(0)
    trainer = Trainer(classifier_output, (ce, pe),
                      sgd(classifier_output.parameters, lr=lr_per_sample),
                      progress_printer)

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200

    for i in range(255):
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)

    evaluation_average = float(trainer.previous_minibatch_evaluation_average)
    loss_average = float(trainer.previous_minibatch_loss_average)
    return evaluation_average, loss_average

if __name__ == '__main__':
    error, _ = train_sequence_classifier()
    print(" error: %f" % error)
