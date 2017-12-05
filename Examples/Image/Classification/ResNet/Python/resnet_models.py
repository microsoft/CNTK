# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk.initializer import he_normal, normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu

#
# assembly components
#
def conv_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), bn_init_scale=1):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False, init_scale=bn_init_scale, disable_regularization=True)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1, 1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init, 1)
    return relu(r)

#
# ResNet components
#
def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3, 3), num_filters)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    p = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2, 2)):
    c1 = conv_bn_relu(input, (3, 3), num_filters, strides)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    s = conv_bn(input, (1, 1), num_filters, strides) # Shortcut
    p = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_basic(l, num_filters)
    return l

def resnet_bottleneck(input, out_num_filters, inter_out_num_filters):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    p = c3 + input
    return relu(p)

def resnet_bottleneck_inc(input, out_num_filters, inter_out_num_filters, stride1x1, stride3x3):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters, strides=stride1x1)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters, strides=stride3x3)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    stride = np.multiply(stride1x1, stride3x3)
    s = conv_bn(input, (1, 1), out_num_filters, strides=stride) # Shortcut
    p = c3 + s
    return relu(p)

def resnet_bottleneck_stack(input, num_stack_layers, out_num_filters, inter_out_num_filters): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_bottleneck(l, out_num_filters, inter_out_num_filters)
    return l

#
# Defines the residual network model for classifying images
#
def create_cifar10_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]

    conv = conv_bn_relu(input, (3, 3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8), name='final_avg_pooling')(r3_2)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z

def create_imagenet_model_basic(input, num_stack_layers, num_classes):
    c_map = [64, 128, 256, 512]

    conv = conv_bn_relu(input, (7, 7), c_map[0], strides=(2, 2))
    pool1 = MaxPooling((3, 3), strides=(2, 2), pad=True)(conv)
    r1 = resnet_basic_stack(pool1, num_stack_layers[0], c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers[1], c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers[2], c_map[2])

    r4_1 = resnet_basic_inc(r3_2, c_map[3])
    r4_2 = resnet_basic_stack(r4_1, num_stack_layers[3], c_map[3])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(7, 7), name='final_avg_pooling')(r4_2)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z

def create_imagenet_model_bottleneck(input, num_stack_layers, num_classes, stride1x1, stride3x3):
    c_map = [64, 128, 256, 512, 1024, 2048]

    # conv1 and max pooling
    conv1 = conv_bn_relu(input, (7, 7), c_map[0], strides=(2, 2))
    pool1 = MaxPooling((3,3), strides=(2,2), pad=True)(conv1)

    # conv2_x
    r2_1 = resnet_bottleneck_inc(pool1, c_map[2], c_map[0], (1, 1), (1, 1))
    r2_2 = resnet_bottleneck_stack(r2_1, num_stack_layers[0], c_map[2], c_map[0])

    # conv3_x
    r3_1 = resnet_bottleneck_inc(r2_2, c_map[3], c_map[1], stride1x1, stride3x3)
    r3_2 = resnet_bottleneck_stack(r3_1, num_stack_layers[1], c_map[3], c_map[1])

    # conv4_x
    r4_1 = resnet_bottleneck_inc(r3_2, c_map[4], c_map[2], stride1x1, stride3x3)
    r4_2 = resnet_bottleneck_stack(r4_1, num_stack_layers[2], c_map[4], c_map[2])

    # conv5_x
    r5_1 = resnet_bottleneck_inc(r4_2, c_map[5], c_map[3], stride1x1, stride3x3)
    r5_2 = resnet_bottleneck_stack(r5_1, num_stack_layers[3], c_map[5], c_map[3])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(7, 7), name='final_avg_pooling')(r5_2)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z
