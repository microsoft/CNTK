# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, splice

def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1,1), pad=True, bnTimeConst=4096, init=he_normal()):
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

def inception_block_with_avgpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out
    
def inception_block_with_maxpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # Max Pooling
    branchPool_maxpool = MaxPooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_maxpool, numPool, (1,1), (1,1), True, bnTimeConst)
    
    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

def inception_block_pass_through(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    
    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (2,2), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (2,2), True, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=True)(input)

    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

def bn_inception_model(input, labelDim, bnTimeConst):

    # 224 x 224 x 3
    conv1 = conv_bn_relu_layer(input, 64, (7,7), (2,2), True, bnTimeConst)
    # 112 x 112 x 64
    pool1 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=True)(conv1)
    # 56 x 56 x 64
    conv2a = conv_bn_relu_layer(pool1, 64, (1,1), (1,1), True, bnTimeConst)
    # 56 x 56 x 64
    conv2b = conv_bn_relu_layer(conv2a, 192, (3,3), (1,1), True, bnTimeConst)
    # 56 x 56 x 192
    pool2 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=True)(conv2b)
    
    # Inception Blocks
    # 28 x 28 x 192
    inception3a = inception_block_with_avgpool(pool2, 64, 64, 64, 64, 96, 32, bnTimeConst)
    # 28 x 28 x 256
    inception3b = inception_block_with_avgpool(inception3a, 64, 64, 96, 64, 96, 64, bnTimeConst) 
    # 28 x 28 x 320
    inception3c = inception_block_pass_through(inception3b, 0, 128, 160, 64, 96, 0, bnTimeConst) 
    # 14 x 14 x 576
    inception4a = inception_block_with_avgpool(inception3c, 224, 64, 96, 96, 128, 128, bnTimeConst) 
    # 14 x 14 x 576
    inception4b = inception_block_with_avgpool(inception4a, 192, 96, 128, 96, 128, 128, bnTimeConst) 
    # 14 x 14 x 576
    inception4c = inception_block_with_avgpool(inception4b, 160, 128, 160, 128, 160, 128, bnTimeConst) 
    # 14 x 14 x 576
    inception4d = inception_block_with_avgpool(inception4c, 96, 128, 192, 160, 192, 128, bnTimeConst) 
    # 14 x 14 x 576
    inception4e = inception_block_pass_through(inception4d, 0, 128, 192, 192, 256, 0, bnTimeConst)
    # 7 x 7 x 1024
    inception5a = inception_block_with_avgpool(inception4e, 352, 192, 320, 160, 224, 128, bnTimeConst) 
    # 7 x 7 x 1024
    inception5b = inception_block_with_maxpool(inception5a, 352, 192, 320, 192, 224, 128, bnTimeConst) 
    
    # Global Average
    # 7 x 7 x 1024
    pool3 = AveragePooling(filter_shape=(7,7))(inception5b)
    # 1 x 1 x 1024
    z = Dense(labelDim, init=he_normal())(pool3)

    return z

def bn_inception_cifar_model(input, labelDim, bnTimeConst):

    # 32 x 32 x 3
    conv1a = conv_bn_relu_layer(input, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1b = conv_bn_relu_layer(conv1a, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1c = conv_bn_relu_layer(conv1b, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv2a = conv_bn_relu_layer(conv1c, 32, (1,1), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv2b = conv_bn_relu_layer(conv2a, 64, (3,3), (1,1), True, bnTimeConst)
    
    # Inception Blocks
    # 32 x 32 x 64
    inception3a = inception_block_with_avgpool(conv2b, 32, 32, 32, 32, 48, 16, bnTimeConst)
    # 32 x 32 x 128
    inception3b = inception_block_pass_through(inception3a, 0, 64, 80, 32, 48, 0, bnTimeConst) 
    # 16 x 16 x 256
    inception4a = inception_block_with_avgpool(inception3b, 96, 48, 64, 48, 64, 64, bnTimeConst) 
    # 16 x 16 x 288
    inception4b = inception_block_with_avgpool(inception4a, 48, 64, 96, 80, 96, 64, bnTimeConst) 
    # 16 x 16 x 288
    inception4c = inception_block_pass_through(inception4b, 0, 128, 192, 192, 256, 0, bnTimeConst)
    # 8 x 8 x 512
    inception5a = inception_block_with_maxpool(inception4c, 176, 96, 160, 96, 112, 64, bnTimeConst) 
    
    # Global Average
    # 8 x 8 x 512
    pool1 = AveragePooling(filter_shape=(8,8))(inception5a)
    # 1 x 1 x 512
    z = Dense(labelDim, init=he_normal())(pool1)

    return z