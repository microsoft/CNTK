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
    times_param = parameter(shape=(input_dim,output_dim))    
    t = times(input,times_param)
    plus_param = parameter(shape=(output_dim,))
    return plus(plus_param,t.output())    

def fully_connected_layer(input, output_dim, device_id, nonlinearity):        
    p = fully_connected_linear_layer(input, output_dim, device_id)
    return nonlinearity(p.output());

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, device, nonlinearity):
    classifier_root = fully_connected_layer(input, hidden_layer_dim, device, nonlinearity)
    for i in range(1, num_hidden_layers):
        classifier_root = fully_connected_layer(classifier_root.output(), hidden_layer_dim, device, nonlinearity)
    
    output_times_param = parameter(shape=(hidden_layer_dim,num_output_classes))
    output_plus_param = parameter(shape=(num_output_classes,))
    t = times(classifier_root.output(),output_times_param)
    classifier_root = plus(output_plus_param,t.output()) 
    return classifier_root;

def conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device):
    num_in_channels = input.shape().dimensions()[0]        
    #TODO: use RandomNormal to initialize, needs to be exposed in the python api
    conv_params = parameter(shape=(num_in_channels, kernel_height, kernel_width, out_feature_map_count), device_id=device)       
    conv_func = convolution(conv_params, input, (num_in_channels, v_stride, h_stride))    
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count,), device_id=device)   
    scale_params = parameter(shape=(out_feature_map_count,), device_id=device)   
    running_mean = constant((out_feature_map_count,), 0.0, device_id=device)
    running_invstd = constant((out_feature_map_count,), 0.0, device_id=device)
    return batch_normalization(conv_func.output(), scale_params, bias_params, running_mean, running_invstd, True, bn_time_const, 0.0, 0.000000001)    

def conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device):
    conv_bn_function = conv_bn_layer(input, out_feature_map_count, kernel_width, kernel_height, h_stride, v_stride, w_scale, b_value, sc_value, bn_time_const, device)
    return relu(conv_bn_function.output())

def resnet_node2(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, device):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device)
    c2 =  conv_bn_layer(c1.output(), out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device)
    p = plus(c2.output(), input)
    return relu(p.output())

def proj_layer(w_proj, input, h_stride, v_stride, b_value, sc_value, bn_time_const, device):
    out_feature_map_count = w_proj.shape().dimensions()[-1];
    #TODO: initialize using b_value and sc_value, needs to be exposed in the python api
    bias_params = parameter(shape=(out_feature_map_count,), device_id=device)   
    scale_params = parameter(shape=(out_feature_map_count,), device_id=device)   
    running_mean = constant((out_feature_map_count,), 0.0, device_id=device)
    running_invstd = constant((out_feature_map_count,), 0.0, device_id=device)
    num_in_channels = input.shape().dimensions()[0]        
    conv_func = convolution(w_proj, input, (num_in_channels, v_stride, h_stride))    
    return batch_normalization(conv_func.output(), scale_params, bias_params, running_mean, running_invstd, True, bn_time_const)

def resnet_node2_inc(input, out_feature_map_count, kernel_width, kernel_height, w_scale, b_value, sc_value, bn_time_const, w_proj, device):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, kernel_width, kernel_height, 2, 2, w_scale, b_value, sc_value, bn_time_const, device)
    c2 =  conv_bn_layer(c1.output(), out_feature_map_count, kernel_width, kernel_height, 1, 1, w_scale, b_value, sc_value, bn_time_const, device)

    c_proj = proj_layer(w_proj, input, 2, 2, b_value, sc_value, bn_time_const, device)
    p = plus(c2.output(), c_proj.output())
    return relu(p.output())

def embedding(input, embedding_dim, device):
    input_dim = input.shape()[0];

    embedding_parameters = parameter(shape=(input_dim, embedding_dim), device_id=device)
    return times(input, embedding_parameters)

def select_last(operand):
    return slice(operand, axis.default_dynamic_axis, -1, 0)

def LSTMP_cell_with_self_stabilization(input, prev_output, prev_cell_state, device):
    input_dim = input.shape()[0]
    output_dim = prev_output.shape()[0];
    cell_dim = prev_cell_state.shape()[0];

    Wxo = parameter(shape=(input_dim, cell_dim), device_id=device)
    Wxi = parameter(shape=(input_dim, cell_dim), device_id=device)
    Wxf = parameter(shape=(input_dim, cell_dim), device_id=device)
    Wxc = parameter(shape=(input_dim, cell_dim), device_id=device)

    Bo = parameter(shape=(cell_dim,), value=0, device_id=device)
    Bc = parameter(shape=(cell_dim,), value=0, device_id=device)
    Bi = parameter(shape=(cell_dim,), value=0, device_id=device)
    Bf = parameter(shape=(cell_dim,), value=0, device_id=device)

    Whi = parameter(shape=(output_dim, cell_dim), device_id=device)
    Wci = parameter(shape=(cell_dim,), device_id=device)

    Whf = parameter(shape=(output_dim, cell_dim), device_id=device)
    Wcf = parameter(shape=(cell_dim,), device_id=device)

    Who = parameter(shape=(output_dim, cell_dim), device_id=device)
    Wco = parameter(shape=(cell_dim,), device_id=device)

    Whc = parameter(shape=(output_dim, cell_dim), device_id=device)

    Wmr = parameter(shape=(cell_dim, output_dim), device_id=device)

    # Stabilization by routing input through an extra scalar parameter
    sWxo = parameter(shape=(), value=0, device_id=device)
    sWxi = parameter(shape=(), value=0, device_id=device)
    sWxf = parameter(shape=(), value=0, device_id=device)
    sWxc = parameter(shape=(), value=0, device_id=device)

    sWhi = parameter(shape=(), value=0, device_id=device)
    sWci = parameter(shape=(), value=0, device_id=device)

    sWhf = parameter(shape=(), value=0, device_id=device)
    sWcf = parameter(shape=(), value=0, device_id=device)
    sWho = parameter(shape=(), value=0, device_id=device)
    sWco = parameter(shape=(), value=0, device_id=device)
    sWhc = parameter(shape=(), value=0, device_id=device)

    sWmr = parameter(shape=(), value=0, device_id=device)

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

    temp1 = element_times(expsWxi.output(), input)
    Wxix = times(temp1.output(), Wxi)    
    Whidh = times(element_times(expsWhi.output(), prev_output).output(), Whi)
    Wcidc = element_times(Wci, element_times(expsWci.output(), prev_cell_state).output())

    #TODO: use (+) instead of plus, but we need to wrap output() so it return our inhereted classes    
    it = sigmoid(plus(plus(Wxix.output(),Bi).output(),plus(Whidh.output(),Wcidc.output()).output()).output())
    Wxcx = times(element_times(expsWxc.output(), input).output(), Wxc)
    Whcdh = times(element_times(expsWhc.output(), prev_output).output(), Whc)
    bit = element_times(it.output(), tanh(plus(plus(Wxcx.output(), Whcdh.output()).output(), Bc).output()).output())
    Wxfx = times(element_times(expsWxf.output(), input).output(), Wxf)
    Whfdh = times(element_times(expsWhf.output(), prev_output).output(), Whf)
    Wcfdc = element_times(Wcf, element_times(expsWcf.output(), prev_cell_state).output())
    
    #TODO: use (+) instead of plus, but we need to wrap output() so it return our inherited classes
    ft = sigmoid(plus(plus(Wxfx.output(),Bf).output(),plus(Whfdh.output(),Wcfdc.output()).output()).output())
    bft = element_times(ft.output(), prev_cell_state)

    ct = plus(bft.output(), bit.output())

    Wxox = times(element_times(expsWxo.output(), input).output(), Wxo)
    Whodh = times(element_times(expsWho.output(), prev_output).output(), Who)
    Wcoct = element_times(Wco, element_times(expsWco.output(), ct.output()).output())

    #TODO: use (+) instead of plus, but we need to wrap output() so it return our inherited classes
    ot = sigmoid(plus(plus(Wxox.output(),Bo).output(),plus(Whodh.output(),Wcoct.output()).output()).output())

    mt = element_times(ot.output(), tanh(ct.output()).output())    
    return (times(element_times(expsWmr.output(), mt.output()).output(), Wmr), ct)


def LSTMP_component_with_self_stabilization(input, output_dim, cell_dim, device):
    dh = placeholder(shape=(output_dim,))
    dc = placeholder(shape=(cell_dim,))

    LSTMCell = LSTMP_cell_with_self_stabilization(input, dh, dc, device);
    
    actualDh = past_value(constant((), 0.0, device_id=device), LSTMCell[0].output(), 1);
    actualDc = past_value(constant((), 0.0, device_id=device), LSTMCell[1].output(), 1);

    # Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    return LSTMCell[0].replace_placeholders({ dh : actualDh.output(), dc : actualDc.output()})

