# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.ops import *

def fully_connected_layer(input, output_dim, device_id, nonlinearity):        
    input_dim = input.shape()[0]    
    times_param = parameter(shape=(input_dim,output_dim))    
    t = times(input,times_param)
    plus_param = parameter(shape=(output_dim,))
    p = plus(plus_param,t.output())    
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