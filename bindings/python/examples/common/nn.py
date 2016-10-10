# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.ops import *
from cntk.utils import sanitize_dtype_cntk, get_train_eval_criterion, get_train_loss
from cntk.initializer import glorot_uniform


def linear_layer(input_var, output_dim):
    shape = input_var.shape()

    input_dim = shape[0]
    times_param = parameter(shape=(input_dim, output_dim), init=glorot_uniform())
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


def conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const):
    shape = input.shape()
    num_in_channels = shape[0]
    #TODO: use RandomNormal to initialize, needs to be exposed in the python api
    conv_params = parameter(shape=(out_feature_map_count, num_in_channels, kernel_height, kernel_width), init=glorot_uniform(output_rank=-1, filter_rank=2))
    conv_func = convolution(conv_params, input, (num_in_channels, v_stride, h_stride))

    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count), init=b_value)
    scale_params = parameter(shape=(out_feature_map_count), init=sc_value)
    running_mean = constant((out_feature_map_count), 0.0)
    running_invstd = constant((out_feature_map_count), 0.0)
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const, 0.0, 0.000000001)


def conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const):
    conv_bn_function = conv_bn_layer(input, out_feature_map_count, kernel_width,
                                     kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const)
    return relu(conv_bn_function)


def resnet_node2(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width,
                            kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, kernel_width,
                       kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)
    p = c2 + input
    return relu(p)


def proj_layer(w_proj, input, h_stride, v_stride, b_value, sc_value, bn_time_const):
    shape = input.shape()
    num_in_channels = shape[0]
    conv_func = convolution(w_proj, input, (num_in_channels, v_stride, h_stride))
    out_feature_map_count = w_proj.shape()[-1];
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count), init=b_value)
    scale_params = parameter(shape=(out_feature_map_count), init=sc_value)
    running_mean = constant((out_feature_map_count), 0.0)
    running_invstd = constant((out_feature_map_count), 0.0)
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const)


def resnet_node2_inc(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, w_proj):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width,
                            kernel_height, 2, 2, w_scale, b_value, sc_value, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, kernel_width,
                       kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const)

    c_proj = proj_layer(w_proj, input, 2, 2, b_value, sc_value, bn_time_const)
    p = c2 + c_proj
    return relu(p)


def embedding(input, embedding_dim):
    input_dim = input.shape()[0]

    embedding_parameters = parameter(shape=(input_dim, embedding_dim), init=glorot_uniform())
    return times(input, embedding_parameters)


def select_last(operand):
    return slice(operand, Axis.default_dynamic_axis(), -1, 0)


def stabilize(operand):
    scalar_constant = 4.0
    f = constant(sanitize_dtype_cntk(np.float32), scalar_constant)
    fInv = constant(sanitize_dtype_cntk(np.float32), 1.0 / scalar_constant)

    beta = element_times(fInv, log(constant(sanitize_dtype_cntk(
        np.float32), 1.0) + exp(element_times(f, parameter(init=0.99537863)))))
    return element_times(beta, operand)


def LSTMP_cell_with_self_stabilization(input, prev_output, prev_cell_state):
    input_dim = input.shape()[0]
    output_dim = prev_output.shape()[0]
    cell_dim = prev_cell_state.shape()[0]

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
    dh = placeholder_variable(
        shape=(output_dim), dynamic_axes=input.dynamic_axes())
    dc = placeholder_variable(
        shape=(cell_dim), dynamic_axes=input.dynamic_axes())

    LSTMCell = LSTMP_cell_with_self_stabilization(input, dh, dc)
    actualDh = recurrence_hookH(LSTMCell[0])
    actualDc = recurrence_hookC(LSTMCell[1])

    # Form the recurrence loop by replacing the dh and dc placeholders with
    # the actualDh and actualDc
    LSTMCell[0].replace_placeholders(
        {dh: actualDh.output(), dc: actualDc.output()})
    
    return (LSTMCell[0], LSTMCell[1])


def print_training_progress(trainer, mb, frequency):

    if mb % frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            mb, training_loss, eval_crit))
