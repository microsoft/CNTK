import sys
import os

import numpy as np
import cntk as ct
from cntk.internal import _as_tuple

#
# Porting https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py to CNTK and add
# some extra primitives and wrappers.
#

global_init = ct.normal(0.05) # paper 0.05

def maximum(l, r):
    return ct.element_select(ct.greater(l, r), l, r)

def minimum(l, r):
    return ct.element_select(ct.less(l, r), l, r)

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    return ct.elu(ct.splice(x, -x, axis=0))

def log_sum_exp(x, axis=None):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    rank = len(x.shape)
    m = ct.reshape(ct.reduce_max(x, axis), shape=x.shape[0:axis]+x.shape[axis+1:rank])
    m2 = ct.reduce_max(x, axis)
    return m + ct.reshape(ct.log(ct.reduce_sum(ct.exp(x-m2), axis)), shape=m.shape)

def log_prob_from_logits(x, axis=None):
    """ numerically stable log_softmax implementation that prevents overflow """
    m = ct.reduce_max(x, axis)
    return x - m - ct.log(ct.reduce_sum(ct.exp(x-m), axis=axis))

def l2_normalize(x, axis=None, epsilon=1e-12):
    return x / ct.sqrt(maximum(ct.reduce_sum(x*x, axis=axis), epsilon))

def moments(x, axes=None, shift=None):
    ''' Ported from tensorflow '''
    _axes = _as_tuple(axes)
    if shift is None:
        shift = x
        # Compute true mean while keeping the dims for proper broadcasting.
        for axis in _axes:
            shift = ct.stop_gradient(ct.reduce_mean(shift, axis=axis))

    shifted_mean = x - shift
    for axis in _axes:
        shifted_mean = ct.reduce_mean(shifted_mean, axis=axis)

    variance_mean = (x-shift)*(x-shift)
    for axis in _axes:
        variance_mean = ct.reduce_mean(variance_mean, axis=axis)

    variance = variance_mean - shifted_mean*shifted_mean
    mean = shifted_mean + shift

    return mean, variance

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

global_parameters = {}

def get_parameter(scope, name):
    return global_parameters[scope+'/'+name]

def set_parameter(scope, name, var):
    global_parameters[scope+'/'+name] = var

def get_parameters(scope, names):
    vars = []
    for name in names:
        vars.append(get_parameter(scope, name))
    return vars

def bnorm(input, num_filters):
    ''' Batchnormalization layer. '''

    output_channels_shape = _as_tuple(num_filters)

    # Batchnormalization
    bias_params    = ct.parameter(shape=output_channels_shape, init=0)
    scale_params   = ct.parameter(shape=output_channels_shape, init=1)
    running_mean   = ct.constant(0., output_channels_shape)
    running_invstd = ct.constant(0., output_channels_shape)
    running_count  = ct.constant(0., (1))
    return ct.batch_normalization(input,
                                  scale_params, 
                                  bias_params, 
                                  running_mean, 
                                  running_invstd, 
                                  running_count=running_count, 
                                  spatial=True,
                                  normalization_time_constant=4096, 
                                  use_cudnn_engine=True)

def dense(x, num_units, nonlinearity=None, init=global_init, init_scale=1., counters={}, first_run=False):
    ''' Dense layer. '''

    scope = get_name('dense', counters)
    x_shape  = x.shape

    if first_run:
        V = ct.parameter((num_units, x_shape[0]), init=init, name='V'); set_parameter(scope, 'V', V)
        g = ct.parameter((num_units, 1.), init=init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter((num_units, 1.), name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = V / ct.sqrt(maximum(ct.reduce_sum(V*V, axis=1), 1e-12))        
        x_init = ct.times(V_norm, x)

        m_init, v_init = moments(x_init, axes=(1, ct.Axis.default_batch_axis()))
        scale_init = init_scale / ct.sqrt(v_init + 1e-10)
        ct.assign(g, scale_init)
        ct.assign(b, -m_init*scale_init)
        linear = ct.reshape(scale_init, (num_units, 1.)) * (x_init - ct.reshape(m_init, (num_units, 1.)))
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        linear = ct.times(V, x)
        scaler = g / ct.sqrt(maximum(ct.reduce_sum(ct.square(V), axis=1), 1e-12))

        linear = ct.reshape(scaler, (num_units, 1.)) * linear + ct.reshape(b, (num_units, 1.))

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def conv2d(x, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=global_init, init_scale=1., counters={}, first_run=False):
    ''' Convolution layer. '''

    scope = get_name('conv2d', counters)
    output_channels_shape = _as_tuple(num_filters)
    x_channels_shape  = _as_tuple(x.shape[0])

    if first_run:
        V = ct.parameter(output_channels_shape + x_channels_shape + filter_shape, init=init, name='V'); set_parameter(scope, 'V', V)
        g = ct.parameter(output_channels_shape + (1,) * len(filter_shape), init=init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter(output_channels_shape + (1,) * len(filter_shape), name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = V / ct.sqrt(maximum(ct.reduce_sum(ct.reduce_sum(ct.reduce_sum(V*V, axis=0), axis=2), axis=3), 1e-12))        
        x_init = ct.convolution(V_norm, x, strides=x_channels_shape + strides, auto_padding=_as_tuple(pad))

        m_init, v_init = moments(x_init, axes=(1,2,ct.Axis.default_batch_axis()))
        scale_init = init_scale / ct.sqrt(v_init + 1e-8)
        ct.assign(g, scale_init)
        ct.assign(b, -m_init*scale_init)

        linear = ct.reshape(scale_init,(num_filters, 1,1))*(x_init-ct.reshape(m_init,(num_filters, 1,1)))
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = V / ct.sqrt(maximum(ct.reduce_sum(ct.reduce_sum(ct.reduce_sum(V*V, axis=0), axis=2), axis=3), 1e-12))
        W = ct.reshape(g,(num_filters,1,1,1)) * V_norm

        linear = ct.convolution(W, x, strides=x_channels_shape + strides, auto_padding=_as_tuple(pad)) + b

    # Batchnormalization
    # linear = bnorm(linear, num_filters)

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def deconv2d(x, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=global_init, init_scale=1., counters={}, first_run=False):
    ''' Deconvolution layer. '''

    scope = get_name('deconv2d', counters)    
    output_channels_shape = _as_tuple(num_filters)
    x_shape               = x.shape # CHW
    x_channels_shape      = _as_tuple(x.shape[0])

    if pad:
        output_shape = (num_filters, x_shape[1] * strides[0], x_shape[2] * strides[1])
    else:
        output_shape = (num_filters, x_shape[1] * strides[0] + filter_shape[0] - 1, x_shape[2] * strides[1] + filter_shape[1] - 1)

    if first_run:
        V = ct.parameter(x_channels_shape + output_channels_shape + filter_shape, init=init, name='V'); set_parameter(scope, 'V', V)
        g = ct.parameter(output_channels_shape + (1,) * len(filter_shape), init=init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter(output_channels_shape + (1,) * len(filter_shape), name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = V / ct.sqrt(maximum(ct.reduce_sum(ct.reduce_sum(ct.reduce_sum(V*V, axis=0), axis=2), axis=3), 1e-12))        
        x_init = ct.convolution_transpose(V_norm, x, strides=x_channels_shape + strides, output_shape=output_shape, auto_padding=_as_tuple(pad))

        m_init, v_init = moments(x_init, axes=(1,2,ct.Axis.default_batch_axis()))
        scale_init = init_scale / ct.sqrt(v_init + 1e-8)
        ct.assign(g, scale_init)
        ct.assign(b, -m_init*scale_init)

        linear = ct.reshape(scale_init, (num_filters,1,1))*(x_init-ct.reshape(m_init, (num_filters,1,1)))
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = V / ct.sqrt(maximum(ct.reduce_sum(ct.reduce_sum(ct.reduce_sum(V*V, axis=0), axis=2), axis=3), 1e-12))
        W = ct.reshape(g,(1,num_filters,1,1)) * V_norm

        linear = ct.convolution_transpose(W, x, strides=x_channels_shape + strides, output_shape=output_shape, auto_padding=_as_tuple(pad)) + b

    # Batchnormalization
    # linear = bnorm(linear, num_filters)

    if nonlinearity == None:
        return linear

    return nonlinearity(linear)

def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s  = x.shape

    x = ct.reshape(x, (s[0], np.prod(s[1:])))
    x = dense(x, num_units, **kwargs)
    return ct.reshape(x, (num_units,)+s[1:])

def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=global_init, dropout_p=0., counters={}, first_run=False):
    xs = x.shape
    num_filters = xs[0]

    c1 = conv(nonlinearity(x), num_filters, counters=counters, first_run=first_run)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        ashape = a.shape
        c1s = c1.shape
        c1 += nin(nonlinearity(a), num_filters, counters=counters, first_run=first_run)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = ct.dropout(c1, dropout_p)
    c2 = conv(c1, num_filters * 2, counters=counters, first_run=first_run)

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
    return x[:,:(xs[1]-filter_shape[0]+1),int((filter_shape[1]-1)/2):(xs[2]-int((filter_shape[1]-1)/2))]

def down_right_shifted_conv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    xs = x.shape
    x = ct.splice(ct.constant(value=0., shape=(xs[0],filter_shape[0]-1,xs[2])), x, axis=1) if filter_shape[0] > 1 else x; xs = x.shape
    x = ct.splice(ct.constant(value=0., shape=(xs[0],xs[1],filter_shape[1]-1)), x, axis=2) if filter_shape[1] > 1 else x
    return conv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)

def down_right_shifted_deconv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    x = deconv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    xs = x.shape
    return x[:,:(xs[1]-filter_shape[0]+1):,:(xs[2]-filter_shape[1]+1)]
