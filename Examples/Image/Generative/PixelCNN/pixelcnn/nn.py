import sys
import os

import numpy as np
import cntk as ct
from cntk.utils import _as_tuple

#
# Porting https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py to CNTK and add
# some extra primitives and wrappers.
#

def maximum(l, r):
    return ct.element_select(ct.greater(l, r), l, r)

def minimum(l, r):
    return ct.element_select(ct.less(l, r), l, r)

def l2_normalize(x, axis=None):
    return x / ct.sqrt(ct.reduce_sum(x*x, axis=axis))

def softplus(x):
    return ct.log(ct.exp(x) + 1)

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    return ct.elu(ct.splice(x, -x, axis=0))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = ct.reshape(ct.reduce_max(x, axis), shape=x.shape[0:axis])
    m2 = ct.reduce_max(x, axis)
    return m + ct.reshape(ct.log(ct.reduce_sum(ct.exp(x-m2), axis)), shape=m.shape)

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = ct.reduce_max(x, axis)
    return x - m - ct.log(ct.reduce_sum(ct.exp(x-m), axis=axis))

def dense(input, num_units, nonlinearity = None, init=ct.normal(0.05)):
    input_shape  = input.shape            # (3, HW)
    output_shape = _as_tuple(num_units)

    b = ct.parameter(output_shape+(input_shape[1],), name='b')
    W = ct.parameter(output_shape+(input_shape[0],), init=init, name='W')  # (n,3)

    linear = b + ct.times(W, input)  # (n, HW)
    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def conv2d(input, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=ct.normal(0.05)):
    '''
    Convolution layer.
    '''
    output_channels_shape = _as_tuple(num_filters)
    input_channels_shape  = _as_tuple(input.shape[0])

    W = ct.parameter(output_channels_shape + input_channels_shape + filter_shape, init=init, name='W')
    b = ct.parameter(output_channels_shape + (1,) * len(filter_shape), name='b')
    linear = ct.convolution(W, input, strides=input_channels_shape + strides, auto_padding=_as_tuple(pad)) + b

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def deconv2d(input, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=ct.normal(0.05)):
    '''
    Deconvolution layer.
    '''
    output_channels_shape = _as_tuple(num_filters)
    input_shape           = input.shape    
    input_channels_shape  = _as_tuple(input.shape[0])

    W = ct.parameter(input_channels_shape + output_channels_shape + filter_shape, init=init, name='W')
    b = ct.parameter(output_channels_shape + (1,) * len(filter_shape), name='b')
    linear = ct.convolution(W, input, strides=input_channels_shape + strides, auto_padding=_as_tuple(pad), transpose=True) + b
    output_shape = linear.shape

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s  = x.shape

    x = ct.reshape(x, (s[0], np.prod(s[1:])))
    x = dense(x, num_units, **kwargs)
    return ct.reshape(x, (num_units,)+s[1:])

def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=ct.normal(0.05), dropout_p=0.):
    xs = x.shape
    num_filters = xs[0]

    c1 = conv(nonlinearity(x), num_filters)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        ashape = a.shape
        c1s = c1.shape
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = ct.dropout(c1, dropout_p)
    c2 = conv(c1, num_filters * 2)

    # add projection of h vector if included: conditional generation
    if h is not None:
       h_shape = h.shape()
       Wh = ct.parameter(h_shape + (2 * num_filters,), init=init, name='Wh')
       c2 = c2 + ct.reshape(ct.times(h, Wc), (2 * num_filters, 1, 1))

    a = c2[:num_filters,:,:]
    b = c2[num_filters:2*num_filters,:,:]
    c3 = a * ct.sigmoid(b)
    return x + c3

''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = x.shape
    # return tf.concat(1,[tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]]) # (B, 32,32,3)  BHWC
    return ct.splice(ct.constant(value=0., shape=(xs[0],1,xs[2])), x[:,:xs[1]-1,:], axis=1) # (3,32,32) CHW

def right_shift(x):
    xs = x.shape
    # return tf.concat(2,[tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]])  # (B, 32,32,3)  BHWC
    return ct.splice(ct.constant(value=0., shape=(xs[0],xs[1],1)), x[:,:,:xs[2]-1], axis=2) # (3,32,32) CHW

def down_shifted_conv2d(x, num_filters, filter_shape=(2,3), strides=(1,1), **kwargs):
    # x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    xs = x.shape
    pad_w = int((filter_shape[1]-1)/2)
    x = ct.splice(ct.constant(value=0., shape=(xs[0],filter_shape[0]-1,xs[2])), x, axis=1) if filter_shape[0] > 1 else x; xs = x.shape
    x = ct.splice(ct.constant(value=0., shape=(xs[0],xs[1],pad_w)), x, axis=2) if pad_w > 0 else x
    x = ct.splice(x, ct.constant(value=0., shape=(xs[0],xs[1],pad_w)), axis=2) if pad_w > 0 else x
    x = conv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    return x

def down_shifted_deconv2d(x, num_filters, filter_shape=(2,3), strides=(1,1), **kwargs):
    x = deconv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    xs = x.shape
    return x[:,:,0:(xs[2]-int((filter_shape[1]-1)/2))]    
    #return x[:,:(xs[1]-filter_shape[0]+1),int((filter_shape[1]-1)/2):(xs[2]-int((filter_shape[1]-1)/2))]

def down_right_shifted_conv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    xs = x.shape
    x = ct.splice(ct.constant(value=0., shape=(xs[0],filter_shape[0]-1,xs[2])), x, axis=1) if filter_shape[0] > 1 else x; xs = x.shape
    x = ct.splice(ct.constant(value=0., shape=(xs[0],xs[1],filter_shape[1]-1)), x, axis=2) if filter_shape[1] > 1 else x
    return conv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)

def down_right_shifted_deconv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    x = deconv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    xs = x.shape
    return x #x[:,:(xs[1]-filter_shape[0]+1):,:(xs[2]-filter_shape[1]+1)]
