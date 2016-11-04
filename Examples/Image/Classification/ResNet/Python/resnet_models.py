# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu

#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init) 
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    s  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2 + s
    return relu(p)

def resnet_basic_stack(num, input, num_filters): 
    assert (num >= 0)
    l = input 
    for _ in range(num): 
        l = resnet_basic(l, num_filters)
    return l 

#   
# Defines the residual network model for classifying images
#
def create_resnet20_cifar10_model(input, num_classes):
    c_map = [16, 32, 64]
    numLayers = 3

    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(numLayers, conv, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(numLayers-1, r2_1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(numLayers-1, r3_1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8,8))(r3_2) 
    z = Dense(num_classes)(pool)

    # learning parameters 
    max_epochs = 160
    lr_per_mb = [1.0]*80+[0.1]*40+[0.01]
    momentum = 0.9 
    l2_reg_weight = 0.0001

    return z, max_epochs, lr_per_mb, momentum, l2_reg_weight

def create_resnet110_cifar10_model(input, num_classes):
    c_map = [16, 32, 64]
    numLayers = 18
    
    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(numLayers, conv, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(numLayers-1, r2_1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(numLayers-1, r3_1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8,8))(r3_2) 
    z = Dense(num_classes)(pool)

    # learning parameters 
    max_epochs = 160
    lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    momentum = 0.9 
    l2_reg_weight = 0.0001

    return z, max_epochs, lr_per_mb, momentum, l2_reg_weight
