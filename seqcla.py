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
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import *
from cntk.layers import *

#abs_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(abs_path, "..", "..", "..", "common"))
#from nn import LSTMP_component_with_self_stabilization, embedding, linear_layer, print_training_progress

### BEGIN nn.py
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# Note: the files in the 'common' folder are deprecated and will be replaced by the layer library

import numpy as np
from cntk.ops import *
from cntk.initializer import glorot_uniform, he_normal


def linear_layer(input_var, output_dim):
    times_param = parameter(shape=(list(input_var.shape)+[output_dim]), init=glorot_uniform())
    bias_param = parameter(shape=(output_dim), init=0)

    t = times(input_var, times_param)
    return bias_param + t


def fully_connected_layer(input, output_dim, nonlinearity):
    p = linear_layer(input, output_dim)
    return nonlinearity(p)

# Defines a multilayer feedforward classification model


def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
    r = fully_connected_layer(input, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        r = fully_connected_layer(r, hidden_layer_dim, nonlinearity)

    return linear_layer(r, num_output_classes)


def conv_bn_layer(input, out_feature_map_count, kernel_shape, strides, bn_time_const, b_value=0, sc_value=1):
    num_in_channels = input.shape[0]
    kernel_width = kernel_shape[0]
    kernel_height = kernel_shape[1]
    v_stride = strides[0]
    h_stride = strides[1]
    #TODO: use RandomNormal to initialize, needs to be exposed in the python api
    conv_params = parameter(shape=(out_feature_map_count, num_in_channels, kernel_height, kernel_width), init=he_normal())
    conv_func = convolution(conv_params, input, (num_in_channels, v_stride, h_stride))

    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count), init=b_value)
    scale_params = parameter(shape=(out_feature_map_count), init=sc_value)
    running_mean = constant(0., (out_feature_map_count))
    running_invstd = constant(0., (out_feature_map_count))
    running_count = constant(0., (1))
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, running_count=running_count, spatial=True,
        normalization_time_constant=bn_time_const, use_cudnn_engine=True)


def conv_bn_relu_layer(input, out_feature_map_count, kernel_shape, strides, bn_time_const, b_value=0, sc_value=1):
    conv_bn_function = conv_bn_layer(input, out_feature_map_count, kernel_shape, strides, bn_time_const, b_value, sc_value)
    return relu(conv_bn_function)


def embedding(input, embedding_dim):
    input_dim = input.shape[0]

    embedding_parameters = parameter(shape=(input_dim, embedding_dim), init=glorot_uniform())
    return times(input, embedding_parameters)


def select_last(operand):
    return slice(operand, Axis.default_dynamic_axis(), -1, 0)


def stabilize(operand):
    scalar_constant = 4.0
    f = constant(scalar_constant)
    fInv = constant(1.0 / scalar_constant)

    beta = element_times(fInv, 
            log(1.0 + exp(element_times(f, parameter(init=0.99537863)))))
    return element_times(beta, operand)


def LSTMP_cell_with_self_stabilization(input, prev_output, prev_cell_state):
    input_dim = input.shape[0]
    output_dim = prev_output.shape[0]
    cell_dim = prev_cell_state.shape[0]

    Wxo = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxi = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxf = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())
    Wxc = parameter(shape=(input_dim, cell_dim), init=glorot_uniform())

    Bo = parameter(shape=(cell_dim), init=0)
    Bc = parameter(shape=(cell_dim), init=0)
    Bi = parameter(shape=(cell_dim), init=0)
    Bf = parameter(shape=(cell_dim), init=0)

    Whi = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wci = parameter(shape=(cell_dim), init=glorot_uniform())

    Whf = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wcf = parameter(shape=(cell_dim), init=glorot_uniform())

    Who = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())
    Wco = parameter(shape=(cell_dim), init=glorot_uniform())

    Whc = parameter(shape=(output_dim, cell_dim), init=glorot_uniform())

    Wmr = parameter(shape=(cell_dim, output_dim), init=glorot_uniform())

    # Stabilization by routing input through an extra scalar parameter
    sWxo = parameter(init=0)
    sWxi = parameter(init=0)
    sWxf = parameter(init=0)
    sWxc = parameter(init=0)

    sWhi = parameter(init=0)
    sWci = parameter(init=0)

    sWhf = parameter(init=0)
    sWcf = parameter(init=0)
    sWho = parameter(init=0)
    sWco = parameter(init=0)
    sWhc = parameter(init=0)

    sWmr = parameter(init=0)

    expsWxo = exp(sWxo)
    expsWxi = exp(sWxi)
    expsWxf = exp(sWxf)
    expsWxc = exp(sWxc)

    expsWhi = exp(sWhi)
    expsWci = exp(sWci)

    expsWhf = exp(sWhf)
    expsWcf = exp(sWcf)
    expsWho = exp(sWho)
    expsWco = exp(sWco)
    expsWhc = exp(sWhc)

    expsWmr = exp(sWmr)

    Wxix = times(element_times(expsWxi, input), Wxi)
    Whidh = times(element_times(expsWhi, prev_output), Whi)
    Wcidc = element_times(Wci, element_times(expsWci, prev_cell_state))

    it = sigmoid(Wxix + Bi + Whidh + Wcidc)
    Wxcx = times(element_times(expsWxc, input), Wxc)
    Whcdh = times(element_times(expsWhc, prev_output), Whc)
    bit = element_times(it, tanh(Wxcx + Whcdh + Bc))
    Wxfx = times(element_times(expsWxf, input), Wxf)
    Whfdh = times(element_times(expsWhf, prev_output), Whf)
    Wcfdc = element_times(Wcf, element_times(expsWcf, prev_cell_state))

    ft = sigmoid(Wxfx + Bf + Whfdh + Wcfdc)
    bft = element_times(ft, prev_cell_state)

    ct = bft + bit

    Wxox = times(element_times(expsWxo, input), Wxo)
    Whodh = times(element_times(expsWho, prev_output), Who)
    Wcoct = element_times(Wco, element_times(expsWco, ct))

    ot = sigmoid(Wxox + Bo + Whodh + Wcoct)

    mt = element_times(ot, tanh(ct))
    return (times(element_times(expsWmr, mt), Wmr), ct)


def LSTMP_component_with_self_stabilization(input, output_dim, cell_dim, recurrence_hookH=past_value, recurrence_hookC=past_value):
    dh = placeholder(
        shape=(output_dim), dynamic_axes=input.dynamic_axes)
    #dc = placeholder(
    #    shape=(cell_dim), dynamic_axes=input.dynamic_axes)

    #LSTMCell = LSTMP_cell_with_self_stabilization(input, dh, dc)
    LSTMCell = RNNUnit(cell_dim, activation=relu)(dh, input)
    actualDh = recurrence_hookH(LSTMCell)#[0])
    #actualDc = recurrence_hookC(LSTMCell[1])

    # Form the recurrence loop by replacing the dh and dc placeholders with
    # the actualDh and actualDc
    LSTMCell.replace_placeholders(
        {dh: actualDh.output})#, dc: actualDc.output})
    
    return (LSTMCell)#, LSTMCell[1])


def print_training_progress(trainer, mb, frequency):

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_crit = trainer.previous_minibatch_evaluation_average
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            mb, training_loss, eval_crit))
### END nn.py


# Creates the reader
def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# Defines the LSTM model for classifying sequences
def LSTM_sequence_classifer_net(feature, num_output_classes, embedding_dim, LSTM_dim, cell_dim):
    return Sequential([
        Embedding(embedding_dim),
        Fold(RNNUnit(cell_dim, activation=relu)),
        Dense(num_output_classes)
    ])(feature)
    embedding_function = embedding(feature, embedding_dim)
    LSTM_function = LSTMP_component_with_self_stabilization(
        embedding_function.output, LSTM_dim, cell_dim)  #[0]
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
    features = sequence.input(shape=input_dim, is_sparse=True)
    label = input(num_output_classes)

    # Instantiate the sequence classification model
    model = LSTM_sequence_classifer_net(
        features, num_output_classes, embedding_dim, hidden_dim, cell_dim)

    ce = cross_entropy_with_softmax(model, label)
    pe = classification_error(model, label)

    rel_path = r"../CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    reader = create_reader(path, True, input_dim, num_output_classes)

    lr_per_sample = learning_rate_schedule(0.05, UnitType.sample)
    # Instantiate the trainer object to drive the model training
    trainer = Trainer(model, (ce, pe),
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
        input_map = {
            features : reader.streams.features,
            label    : reader.streams.labels
        }
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
        #print_training_progress(trainer, i, training_progress_output_freq)
        progress_printer.update_with_trainer(trainer, with_metric=True)    # log progress
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

    error, _ = train_sequence_classifier()
    print("Error: %f" % error)
