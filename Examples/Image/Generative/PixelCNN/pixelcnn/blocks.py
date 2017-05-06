import sys
import os

import numpy as np
import cntk as ct

from cntk.internal import _as_tuple

def conv2d(inputs, num_filters, filter_shape, strides=(1,1), pad=True, mask_type=None, bias=True, input_condition=None, nonlinearity=ct.relu, init=ct.glorot_uniform()):
    '''
    Convolution layer with mask and conditional input support.
    '''    
    output_channels_shape = _as_tuple(num_filters)
    input_channels_shape  = _as_tuple(inputs.shape[0])
    kernel_shape = input_channels_shape + filter_shape

    W = ct.parameter(output_channels_shape + kernel_shape, init=init, name='W')

    if mask_type is not None:
        filter_center = (filter_shape[0] // 2, filter_shape[1] // 2)

        mask_shape = output_channels_shape + kernel_shape
        mask = np.ones(mask_shape, dtype=np.float32)
        mask[:,:,filter_center[0]:,filter_center[1]+1:] = 0
        mask[:,:,filter_center[0]+1:,:] = 0.

        if mask_type == 'a':
            mask[:,:,filter_center[0],filter_center[1]] = 0

        W = ct.element_times(W, ct.constant(mask))

    if bias:
        b = ct.parameter(output_channels_shape + (1,) * len(filter_shape), name='b')
        linear = ct.convolution(W, inputs, strides=input_channels_shape + strides, auto_padding=_as_tuple(pad)) + b
    else:
        linear = ct.convolution(W, inputs, strides=input_channels_shape + strides, auto_padding=_as_tuple(pad))

    if input_condition is not None:
       input_condition_shape = input_condition.shape
       Wc = ct.parameter(input_condition_shape + output_channels_shape + (1,) * len(filter_shape), init=init, name='Wc')
       linear = linear + ct.times(input_condition, Wc)

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def residual_block(input, mask_type = None, init=ct.glorot_uniform()):
    '''
    Residual block, from PixelRNN paper (https://arxiv.org/pdf/1601.06759v3.pdf), used
    to build up the hidden layers in PixelCNN.
    '''
    num_filters_2h = input.shape[0]
    num_filters_h  = num_filters_2h // 2

    l = conv2d(input, num_filters_h, (1,1), (1,1), True, mask_type)
    l = conv2d(l, num_filters_h, (3,3), (1,1), True, mask_type)
    l = conv2d(l, num_filters_2h, (1,1), (1,1), True, mask_type)

    return input + l

def gated_conv2d(input, num_filters, filter_shape, strides = (1,1), pad = True, mask_type = None, input_condition=None, init=ct.glorot_uniform()):
    '''
    Basic gated convolution layer from Condition PixelCNN paper (https://arxiv.org/pdf/1606.05328v2.pdf).
    '''
    f = conv2d(input, num_filters, filter_shape, strides, pad, mask_type, False, input_condition, ct.tanh, init)
    g = conv2d(input, num_filters, filter_shape, strides, pad, mask_type, False, input_condition, ct.sigmoid, init)

    return ct.element_times(f, g)

# def conv2d(inputs, num_filters, filter_shape, strides=(1,1), pad=True, mask_type=None, bias=True, input_condition=None, nonlinearity=ct.relu, init=ct.glorot_uniform()):
def gated_residual_block(input_v, input_h, filter_shape, mask_type = None, input_condition=None, init=ct.glorot_uniform()):
    '''
    Condition PixelCNN building block (https://arxiv.org/pdf/1606.05328v2.pdf), it is composed of vertical stack
    and horizontal stack with resnet connection. 
    '''
    num_filters = input_h.shape[0]
    v = conv2d(input_v, 2*num_filters, filter_shape, (1,1), True, mask_type, False, input_condition, None, init)
    h = conv2d(input_h, 2*num_filters, (1, filter_shape[1]), (1,1), True, None, False, None, None, init)
    h = h + conv2d(v, 2*num_filters, (1,1), (1,1), True, None, None, None, None, init)

    # Vertical stack
    v = ct.element_times(ct.tanh(v[:num_filters,:,:]), ct.sigmoid(v[num_filters:2*num_filters,:,:]))

    # Horizontal stack
    h = ct.element_times(ct.tanh(h[:num_filters,:,:]), ct.sigmoid(h[num_filters:2*num_filters,:,:]))
    h = conv2d(h, num_filters, (1,1), (1,1), True, None, False, None, None, init)
    h = h + input_h

    return v, h
