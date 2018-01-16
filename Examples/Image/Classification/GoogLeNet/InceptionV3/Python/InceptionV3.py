# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense, Dropout
from cntk.ops import minus, element_times, relu, splice

#
# This file contains the basic build block of Inception Network as defined in:
#
#   https://arxiv.org/pdf/1512.00567.pdf
#
# and in Tensorflow implementation
#

#
# Convolution layer with Batch Normalization and Rectifier Linear activation.
#
def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1,1), pad=True, bnTimeConst=4096, init=he_normal()):
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn   = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

#
# Figure 5 from https://arxiv.org/pdf/1512.00567.pdf
# Modified with the added 5x5 branch to match Tensorflow implementation
#
def inception_block_1(input, num1x1, num5x5, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 5x5 Convolution
    branch5x5_1 = conv_bn_relu_layer(input, num5x5[0], (1,1), (1,1), True, bnTimeConst)
    branch5x5   = conv_bn_relu_layer(branch5x5_1, num5x5[1], (5,5), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl   = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3,3), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch5x5, branch3x3dbl, branchPool, axis=0)

    return out

def inception_block_2(input, num3x3, num3x3dbl, bnTimeConst):

    # 3x3 Convolution
    branch3x3 = conv_bn_relu_layer(input, num3x3, (3,3), (2,2), False, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl   = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3,3), (2,2), False, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=False)(input)

    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

#
# Figure 6 from https://arxiv.org/pdf/1512.00567.pdf
#
def inception_block_3(input, num1x1, num7x7, num7x7dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 7x7 Convolution
    branch7x7_1 = conv_bn_relu_layer(input, num7x7[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7_2 = conv_bn_relu_layer(branch7x7_1, num7x7[1], (1,7), (1,1), True, bnTimeConst)
    branch7x7   = conv_bn_relu_layer(branch7x7_2, num7x7[2], (7,1), (1,1), True, bnTimeConst)

    # Double 7x7 Convolution
    branch7x7dbl_1 = conv_bn_relu_layer(input, num7x7dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7dbl_2 = conv_bn_relu_layer(branch7x7dbl_1, num7x7dbl[1], (7,1), (1,1), True, bnTimeConst)
    branch7x7dbl_3 = conv_bn_relu_layer(branch7x7dbl_2, num7x7dbl[2], (1,7), (1,1), True, bnTimeConst)
    branch7x7dbl_4 = conv_bn_relu_layer(branch7x7dbl_3, num7x7dbl[3], (7,1), (1,1), True, bnTimeConst)
    branch7x7dbl   = conv_bn_relu_layer(branch7x7dbl_4, num7x7dbl[4], (1,7), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch7x7, branch7x7dbl, branchPool, axis=0)

    return out

def inception_block_4(input, num3x3, num7x7_3x3, bnTimeConst):

    # 3x3 Convolution
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3   = conv_bn_relu_layer(branch3x3_1, num3x3[1], (3,3), (2,2), False, bnTimeConst)

    # 7x7 3x3 Convolution
    branch7x7_3x3_1 = conv_bn_relu_layer(input, num7x7_3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7_3x3_2 = conv_bn_relu_layer(branch7x7_3x3_1, num7x7_3x3[1], (1,7), (1,1), True, bnTimeConst)
    branch7x7_3x3_3 = conv_bn_relu_layer(branch7x7_3x3_2, num7x7_3x3[2], (7,1), (1,1), True, bnTimeConst)
    branch7x7_3x3   = conv_bn_relu_layer(branch7x7_3x3_3, num7x7_3x3[3], (3,3), (2,2), False, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=False)(input)

    out = splice(branch3x3, branch7x7_3x3, branchPool, axis=0)

    return out

#
# Figure 7 from https://arxiv.org/pdf/1512.00567.pdf
#
def inception_block_5(input, num1x1, num3x3, num3x3_3x3, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3 Convolution
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3_2 = conv_bn_relu_layer(branch3x3_1, num3x3[1], (1,3), (1,1), True, bnTimeConst)
    branch3x3_3 = conv_bn_relu_layer(branch3x3_1, num3x3[2], (3,1), (1,1), True, bnTimeConst)
    branch3x3   = splice(branch3x3_2, branch3x3_3, axis=0)

    # 3x3 3x3 Convolution
    branch3x3_3x3_1 = conv_bn_relu_layer(input, num3x3_3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3_3x3_2 = conv_bn_relu_layer(branch3x3_3x3_1, num3x3_3x3[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3_3x3_3 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[1], (1,3), (1,1), True, bnTimeConst)
    branch3x3_3x3_4 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[3], (3,1), (1,1), True, bnTimeConst)
    branch3x3_3x3   = splice(branch3x3_3x3_3, branch3x3_3x3_4, axis=0)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch3x3, branch3x3_3x3, branchPool, axis=0)

    return out


#
# Inception V3 model with normalized input, to use the below function
# remove "ImageNet1K_mean.xml" from each reader.
#
def inception_v3_norm_model(input, labelDim, dropRate, bnTimeConst):

    # Normalize inputs to -1 and 1.
    featMean  = 128
    featScale = 1/128
    input_subtracted = minus(input, featMean)
    input_scaled = element_times(input_subtracted, featScale)

    return inception_v3_model(input_scaled, labelDim, dropRate, bnTimeConst)

#
# Inception V3 model
#
def inception_v3_model(input, labelDim, dropRate, bnTimeConst):

    # 299 x 299 x 3
    conv1 = conv_bn_relu_layer(input, 32, (3,3), (2,2), False, bnTimeConst)
    # 149 x 149 x 32
    conv2 = conv_bn_relu_layer(conv1, 32, (3,3), (1,1), False, bnTimeConst)
    # 147 x 147 x 32
    conv3 = conv_bn_relu_layer(conv2, 64, (3,3), (1,1), True, bnTimeConst)
    # 147 x 147 x 64
    pool1 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=False)(conv3)
    # 73 x 73 x 64
    conv4 = conv_bn_relu_layer(pool1, 80, (1,1), (1,1), False, bnTimeConst)
    # 73 x 73 x 80
    conv5 = conv_bn_relu_layer(conv4, 192, (3,3), (1,1), False, bnTimeConst)
    # 71 x 71 x 192
    pool2 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=False)(conv5)
    # 35 x 35 x 192

    #
    # Inception Blocks
    #
    mixed1 = inception_block_1(pool2, 64, [48, 64], [64, 96, 96], 32, bnTimeConst)
    # 35 x 35 x 256
    mixed2 = inception_block_1(mixed1, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    # 35 x 35 x 288
    mixed3 = inception_block_1(mixed2, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    # 35 x 35 x 288
    mixed4 = inception_block_2(mixed3, 384, [64, 96, 96], bnTimeConst)
    # 17 x 17 x 768
    mixed5 = inception_block_3(mixed4, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed6 = inception_block_3(mixed5, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed7 = inception_block_3(mixed6, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed8 = inception_block_3(mixed7, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed9 = inception_block_4(mixed8, [192, 320], [192, 192, 192, 192], bnTimeConst)
    # 8 x 8 x 1280
    mixed10 = inception_block_5(mixed9, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    # 8 x 8 x 2048
    mixed11 = inception_block_5(mixed10, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    # 8 x 8 x 2048

    #
    # Prediction
    #
    pool3 = AveragePooling(filter_shape=(8,8), pad=False)(mixed11)
    # 1 x 1 x 2048
    drop = Dropout(dropout_rate=dropRate)(pool3)
    # 1 x 1 x 2048
    z = Dense(labelDim, init=he_normal())(drop)

    #
    # Auxiliary
    #
    # 17 x 17 x 768
    auxPool =  AveragePooling(filter_shape=(5,5), strides=(3,3), pad=False)(mixed8)
    # 5 x 5 x 768
    auxConv1 = conv_bn_relu_layer(auxPool, 128, (1,1), (1,1), True, bnTimeConst)
    # 5 x 5 x 128
    auxConv2 = conv_bn_relu_layer(auxConv1, 768, (5,5), (1,1), False, bnTimeConst)
    # 1 x 1 x 768
    aux = Dense(labelDim, init=he_normal())(auxConv2)

    return {
        'z':   z,
        'aux': aux
    }

