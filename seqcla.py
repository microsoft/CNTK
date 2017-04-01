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
from cntk.ops import input, sequence, relu
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import *
from cntk import debugging
from cntk.layers import *
import cntk
import dynamite
import typing

input_dim = 2000
hidden_dim = 25
embedding_dim = 50
num_output_classes = 5

# Create the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# convert CNTK reader's minibatch to our internal representation
def from_cntk_mb(inputs: tuple, variables: tuple):
    def convert(self, var):
        data = self.data
        # unpack MBLayout
        sequences, _ = data.unpack_variable_value(var, True, data.device())
        # turn into correct NDArrayView types
        from cntk.internal.swig_helper import map_if_possible
        has_axis = len(var.dynamic_axes) > 1
        def fix_up(data):
            shape = data.shape().dimensions()  # drop a superfluous length dimension
            item_shape = shape[1:]
            if has_axis:
                # CONTINUE HERE
                # fails with: NDArrayView::SliceView: Cannot create a slice which is not contiguous in memory. This NDArrayView shape = [3 x 2000], slice offset = [0 x 0], slice extent = [2000 x 1].
                #data = [data.slice_view((t,) + (0,) * len(item_shape), (1,) + item_shape) for t in range(shape[0])]
                # BUGBUG: shape parameters are not getting reversed
                data = [data.slice_view(tuple(reversed((t,) + (0,) * len(item_shape))), tuple(reversed((1,) + item_shape))) for t in range(shape[0])]
            else:
                assert shape[0] == 1
                data = data.as_shape(item_shape)
                map_if_possible(data)
            return data
        return [fix_up(seq) for seq in sequences]
    return tuple(convert(inp, var) for inp, var in zip(inputs, variables))

# Define the LSTM model for classifying sequences
cntk.Sequential = cntk.layers.Sequential  # all in one namespace, to use same code for CNTK and dynamite
cntk.Embedding = cntk.layers.Embedding
cntk.Fold = cntk.layers.Fold
cntk.RNNUnit = cntk.layers.RNNUnit
cntk.Dense = cntk.layers.Dense
def create_model(namespace, num_output_classes, embedding_dim, hidden_dim):
    return namespace.Sequential([
        namespace.Embedding(embedding_dim, name='embed'),
        namespace.Fold(namespace.RNNUnit(hidden_dim, activation=namespace.relu, name='rnn')),
        namespace.Dense(num_output_classes, name='dense')
    ])

# define the criterion fnction
# note: not using @Function here since using the same for dynamite
def create_criterion(namespace, model):
    def criterion(input: Sequence[SparseTensor[input_dim]], label: Tensor[num_output_classes]):
        z = model(input)
        ce = namespace.cross_entropy_with_softmax(z, label)
        pe = namespace.classification_error(z, label)
        return (ce, pe)
    return criterion

# Create and train a LSTM sequence classification model
def train(debug_output=False):
    # Input variables denoting the features and label data
    #features = sequence.input(shape=input_dim, is_sparse=True)
    #label = input(num_output_classes)

    # Instantiate the sequence classification model
    model = create_model(cntk, num_output_classes, embedding_dim, hidden_dim)
    dmodel = create_model(dynamite, num_output_classes, embedding_dim, hidden_dim)

    criterion = Function(create_criterion(cntk, model))
    dcriterion = create_criterion(dynamite, dmodel)
    debugging.dump_signature(criterion)

    # transplant parameters
    dmodel.__items__[0].E              .share_data_from(model.embed.E.data)
    dmodel.__items__[1].step_function.W.share_data_from(model.rnn.W  .data)
    dmodel.__items__[1].step_function.R.share_data_from(model.rnn.H  .data)
    dmodel.__items__[1].step_function.b.share_data_from(model.rnn.b  .data)
    dmodel.__items__[2].W              .share_data_from(model.dense.W.data)
    dmodel.__items__[2].b              .share_data_from(model.dense.b.data)

    rel_path = "../CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    reader = create_reader(os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path, True, input_dim, num_output_classes)

    lr_per_sample = learning_rate_schedule(0.05, UnitType.sample)
    # Instantiate the trainer object to drive the model training
    trainer = Trainer(None, criterion,
                      sgd(model.parameters, lr=lr_per_sample))

    # process minibatches and perform model training
    training_progress_output_freq = 10
    log_number_of_parameters(model) ; print()
    progress_printer = ProgressPrinter(freq=training_progress_output_freq, first=10, tag='Training') # more detailed logging
    #progress_printer = ProgressPrinter(tag='Training')

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    for i in range(251):
        mb = reader.next_minibatch(minibatch_size)
        # CNTK static
        trainer.train_minibatch(criterion.argument_map(mb[reader.streams.features], mb[reader.streams.labels]))
        progress_printer.update_with_trainer(trainer, with_metric=True)    # log progress
        # CNTK dynamite
        args = from_cntk_mb((mb[reader.streams.features], mb[reader.streams.labels]), criterion.arguments)
        dynamite.train_minibatch(dcriterion, *args)
        args = None  # deref; otherwise resize will fail
        #print('static', dmodel.__items__[0].E.data.to_ndarray())
        #print('dynamic', model.embed.E.value)
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    import copy

    evaluation_average = copy.copy(
        trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)

    return evaluation_average, loss_average

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())

    train()
