# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.ops import *

def fully_connected_linear_layer(input, output_dim, device_id):
    input_dim = input.shape()[0]
    times_param = parameter(shape=(input_dim,output_dim), device_id=device_id)
    t = times(input,times_param)
    plus_param = parameter(shape=(output_dim,), device_id=device_id)
    return plus_param+t

def fully_connected_layer(input, output_dim, device_id, nonlinearity):
    p = fully_connected_linear_layer(input, output_dim, device_id)
    return nonlinearity(p);

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device_id, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device_id, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root, hidden_layer_dim, device_id, nonlinearity)

    output_times_param = parameter(shape=(hidden_layer_dim,num_output_classes), device_id=device_id)
    output_plus_param = parameter(shape=(num_output_classes,), device_id=device_id)
    t = times(classifier_root,output_times_param)
    return output_plus_param+t

def conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device_id):
    num_in_channels = input.shape().dimensions()[0]
    #TODO: use RandomNormal to initialize, needs to be exposed in the python api
    conv_params = parameter(shape=(num_in_channels, kernel_height, kernel_width, out_feature_map_count), device_id=device_id)
    conv_func = convolution(conv_params, input, (num_in_channels, v_stride, h_stride))
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count,), device_id=device_id)
    scale_params = parameter(shape=(out_feature_map_count,), device_id=device_id)
    running_mean = constant((out_feature_map_count,), 0.0, device_id=device_id)
    running_invstd = constant((out_feature_map_count,), 0.0, device_id=device_id)
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const, 0.0, 0.000000001)

def conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device_id):
    conv_bn_function = conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device_id)
    return relu(conv_bn_function)

def resnet_node2(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, device_id):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device_id)
    c2 =  conv_bn_layer(c1, out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device_id)
    p = c2+input
    return relu(p)

def proj_layer(w_proj, input, h_stride, v_stride, b_value, sc_value, bn_time_const, device_id):
    out_feature_map_count = w_proj.shape().dimensions()[-1];
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count,), device_id=device_id)
    scale_params = parameter(shape=(out_feature_map_count,), device_id=device_id)
    running_mean = constant((out_feature_map_count,), 0.0, device_id=device_id)
    running_invstd = constant((out_feature_map_count,), 0.0, device_id=device_id)
    num_in_channels = input.shape().dimensions()[0]
    conv_func = convolution(w_proj, input, (num_in_channels, v_stride, h_stride))
    return batch_normalization(conv_func, scale_params, bias_params, running_mean, running_invstd, True, bn_time_const)

def resnet_node2_inc(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, w_proj, device_id):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, 2, 2, w_scale, b_value, sc_value, bn_time_const, device_id)
    c2 =  conv_bn_layer(c1, out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device_id)

    c_proj = proj_layer(w_proj, input, 2, 2, b_value, sc_value, bn_time_const, device_id)
    p = c2+c_proj
    return relu(p)

def embedding(input, embedding_dim, device_id):
    input_dim = input.shape()[0];

    embedding_parameters = parameter(shape=(input_dim, embedding_dim), device_id=device_id)
    return times(input, embedding_parameters)

def select_last(operand):
    return slice(operand, Axis.default_dynamic_axis(), -1, 0)

def LSTMP_cell_with_self_stabilization(input, prev_output, prev_cell_state, device_id):
    input_dim = input.shape()[0]
    output_dim = prev_output.shape()[0];
    cell_dim = prev_cell_state.shape()[0];

    Wxo = parameter(shape=(input_dim, cell_dim), device_id=device_id)
    Wxi = parameter(shape=(input_dim, cell_dim), device_id=device_id)
    Wxf = parameter(shape=(input_dim, cell_dim), device_id=device_id)
    Wxc = parameter(shape=(input_dim, cell_dim), device_id=device_id)

    Bo = parameter(shape=(cell_dim,), value=0, device_id=device_id)
    Bc = parameter(shape=(cell_dim,), value=0, device_id=device_id)
    Bi = parameter(shape=(cell_dim,), value=0, device_id=device_id)
    Bf = parameter(shape=(cell_dim,), value=0, device_id=device_id)

    Whi = parameter(shape=(output_dim, cell_dim), device_id=device_id)
    Wci = parameter(shape=(cell_dim,), device_id=device_id)

    Whf = parameter(shape=(output_dim, cell_dim), device_id=device_id)
    Wcf = parameter(shape=(cell_dim,), device_id=device_id)

    Who = parameter(shape=(output_dim, cell_dim), device_id=device_id)
    Wco = parameter(shape=(cell_dim,), device_id=device_id)

    Whc = parameter(shape=(output_dim, cell_dim), device_id=device_id)

    Wmr = parameter(shape=(cell_dim, output_dim), device_id=device_id)

    # Stabilization by routing input through an extra scalar parameter
    sWxo = parameter(shape=(), value=0, device_id=device_id)
    sWxi = parameter(shape=(), value=0, device_id=device_id)
    sWxf = parameter(shape=(), value=0, device_id=device_id)
    sWxc = parameter(shape=(), value=0, device_id=device_id)

    sWhi = parameter(shape=(), value=0, device_id=device_id)
    sWci = parameter(shape=(), value=0, device_id=device_id)

    sWhf = parameter(shape=(), value=0, device_id=device_id)
    sWcf = parameter(shape=(), value=0, device_id=device_id)
    sWho = parameter(shape=(), value=0, device_id=device_id)
    sWco = parameter(shape=(), value=0, device_id=device_id)
    sWhc = parameter(shape=(), value=0, device_id=device_id)

    sWmr = parameter(shape=(), value=0, device_id=device_id)

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

    temp1 = element_times(expsWxi, input)
    Wxix = times(temp1, Wxi)
    Whidh = times(element_times(expsWhi, prev_output), Whi)
    Wcidc = element_times(Wci, element_times(expsWci, prev_cell_state))

    it = sigmoid(Wxix+Bi+Whidh+Wcidc)
    Wxcx = times(element_times(expsWxc, input), Wxc)
    Whcdh = times(element_times(expsWhc, prev_output), Whc)
    bit = element_times(it, tanh(Wxcx+Whcdh+Bc))
    Wxfx = times(element_times(expsWxf, input), Wxf)
    Whfdh = times(element_times(expsWhf, prev_output), Whf)
    Wcfdc = element_times(Wcf, element_times(expsWcf, prev_cell_state))

    ft = sigmoid(Wxfx+Bf+Whfdh+Wcfdc)
    bft = element_times(ft, prev_cell_state)

    ct = bft+bit

    Wxox = times(element_times(expsWxo, input), Wxo)
    Whodh = times(element_times(expsWho, prev_output), Who)
    Wcoct = element_times(Wco, element_times(expsWco, ct))

    ot = sigmoid(Wxox+Bo+Whodh+Wcoct)

    mt = element_times(ot, tanh(ct))
    return (times(element_times(expsWmr, mt), Wmr), ct)


def LSTMP_component_with_self_stabilization(input, output_dim, cell_dim, device_id):
    dh = placeholder(shape=(output_dim,))
    dc = placeholder(shape=(cell_dim,))

    LSTMCell = LSTMP_cell_with_self_stabilization(input, dh, dc, device_id);

    actualDh = past_value(LSTMCell[0], constant((), 0.0, device_id=device_id), 1);
    actualDc = past_value(LSTMCell[1], constant((), 0.0, device_id=device_id), 1);

    # Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    return LSTMCell[0].owner.replace_placeholders({ dh : actualDh, dc : actualDc}).output()

