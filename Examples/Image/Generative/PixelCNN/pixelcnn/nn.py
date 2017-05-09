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
global_g_init = ct.normal(0.05)

def zeros(shape):
    return ct.constant(value=0., shape=shape, dtype=np.float32)

def ones(shape):
    return ct.constant(value=1., shape=shape, dtype=np.float32)

def squeeze(x, axes):
    reduce_axes = tuple()
    new_shape = None
    axes = _as_tuple(axes)
    for axis in axes:
        if np.isscalar(axis):
            reduce_axes += (axis,)

    new_shape = np.squeeze(np.zeros(x.shape), axis=reduce_axes).shape
    return ct.reshape(x, shape=new_shape)

def _one_hot(indices, depth, dtype=np.float32):
    values = np.asarray(indices)
    return np.asarray((np.arange(depth) == values[..., None]), dtype=dtype)

def one_hot(indices, depth, axis=-1, dtype=np.float32):
    ''' Compute one hot from indices similar signature to tensorflow '''
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis = rank + axis + 1

    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,)*len(ls)+depth_range.shape+(1,)*len(rs))
    values = np.reshape(values, ls+(1,)+rs)
    return np.asarray(targets == values, dtype=dtype)

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    return ct.elu(ct.splice(x, -x, axis=0))

def log_sum_exp(x, axis):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m = squeeze(ct.reduce_max(x, axis), axes=axis)
    m2 = ct.reduce_max(x, axis)
    y = ct.exp(x-m2)
    return m + ct.log(squeeze(ct.reduce_sum(y, axis), axes=axis))

def log_prob_from_logits(x, axis):
    """ numerically stable log_softmax implementation that prevents overflow """
    m = ct.reduce_max(x, axis)
    return x - m - ct.log(ct.reduce_sum(ct.exp(x-m), axis=axis))

def moments(x, axes=None, shift=None, keep_dims=False):
    ''' Ported from tensorflow '''
    _axes = _as_tuple(axes)
    if shift is None:
        shift = x
        # Compute true mean while keeping the dims for proper broadcasting.
        for axis in _axes:
            shift = ct.reduce_mean(shift, axis=axis)

    shift = ct.stop_gradient(shift)
    shifted_mean = ct.minus(x, shift)
    for axis in _axes:
        shifted_mean = ct.reduce_mean(shifted_mean, axis=axis)

    variance_mean = ct.square(ct.minus(x,shift))
    for axis in _axes:
        variance_mean = ct.reduce_mean(variance_mean, axis=axis)

    variance = ct.minus(variance_mean, ct.square(shifted_mean))
    mean = ct.plus(shifted_mean, shift)

    if not keep_dims:
        mean = squeeze(mean, axes)
        variance = squeeze(variance, axes)

    return mean, variance

def l2_normalize(x, axes, epsilon=1e-12):
    ''' Ported from tensorflow '''
    _axes = _as_tuple(axes)
    square_sum = ct.square(x)
    for axis in _axes:
        square_sum = ct.reduce_sum(square_sum, axis=axis)
    x_inv_norm = ct.element_divide(1., ct.sqrt(ct.element_max(square_sum, epsilon)))
    return ct.element_times(x, x_inv_norm)

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

def bnorm(x, num_filters):
    ''' Batchnormalization layer. '''

    output_channels_shape = _as_tuple(num_filters)

    # Batchnormalization
    bias_params    = ct.parameter(shape=output_channels_shape, init=0)
    scale_params   = ct.parameter(shape=output_channels_shape, init=1)
    running_mean   = ct.constant(0., output_channels_shape)
    running_invstd = ct.constant(0., output_channels_shape)
    running_count  = ct.constant(0., (1))
    return ct.batch_normalization(x,
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
        g = ct.parameter((num_units,), init=global_g_init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter((num_units,), name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = l2_normalize(V, axes=(1))
        x_init = ct.times(V_norm, x)

        m_init, v_init = moments(x_init, axes=(ct.Axis.default_batch_axis(),1))
        scale_init = init_scale / ct.sqrt(v_init + 1e-10)
        g_new = ct.assign(g, scale_init)
        b_new = ct.assign(b, -m_init*scale_init)

        x_init = ct.reshape(scale_init, (num_units, 1)) * (x_init - ct.reshape(m_init, (num_units, 1))) + ct.reshape(g_new + b_new, (num_units, 1))*0
        if nonlinearity is not None:
            x_init = nonlinearity(x_init)
        return x_init
        
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        x = ct.times(V, x)
        scaler = g / ct.sqrt(squeeze(ct.reduce_sum(ct.square(V), axis=1), axes=1))

        x = ct.reshape(scaler, (num_units, 1)) * x + ct.reshape(b, (num_units, 1))
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def masked_conv2d(x, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, mask_type=None, h=None, bias=True, init=ct.glorot_uniform()):
    ''' Convolution layer with mask and conditional input support. '''
    output_channels_shape = _as_tuple(num_filters)
    x_channels_shape      = _as_tuple(x.shape[0])
    paddings              = (False,) + (pad,)*len(filter_shape)

    W = ct.parameter((num_filters, x.shape[0]) + filter_shape, init=init, name='W')

    if mask_type is not None:
        filter_center = (filter_shape[0] // 2, filter_shape[1] // 2)

        mask = np.ones(W.shape, dtype=np.float32)
        mask[:,:,filter_center[0]:,filter_center[1]+1:] = 0
        mask[:,:,filter_center[0]+1:,:] = 0.

        if mask_type == 'a':
            mask[:,:,filter_center[0],filter_center[1]] = 0

        W = ct.element_times(W, ct.constant(mask))

    if bias:
        b = ct.parameter((num_filters, 1, 1), name='b')        
        x = ct.convolution(W, x, strides=x_channels_shape + strides, auto_padding=paddings) + b
    else:
        x = ct.convolution(W, x, strides=x_channels_shape + strides, auto_padding=paddings)

    if h is not None:
       h_shape = h.shape
       Wc = ct.parameter(h_shape + output_channels_shape + (1,) * len(filter_shape), init=init, name='Wc')
       x = x + ct.times(h, Wc)

    if nonlinearity is not None:
        x = nonlinearity(x)
    return x

def conv2d(x, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=global_init, init_scale=1., counters={}, first_run=False):
    ''' Convolution layer. '''

    scope = get_name('conv2d', counters)
    output_channels_shape = _as_tuple(num_filters)
    x_channels_shape      = _as_tuple(x.shape[0])
    paddings              = (False,) + (pad,)*len(filter_shape)

    if first_run:
        V = ct.parameter(output_channels_shape + x_channels_shape + filter_shape, init=init, name='V'); set_parameter(scope, 'V', V)
        g = ct.parameter(output_channels_shape, init=global_g_init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter(output_channels_shape, name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = l2_normalize(V, axes=(1, 2, 3))
        x_init = ct.convolution(V_norm, x, strides=x_channels_shape + strides, auto_padding=paddings)

        m_init, v_init = moments(x_init, axes=(ct.Axis.default_batch_axis(),1,2))
        scale_init = init_scale / ct.sqrt(v_init + 1e-8)
        g_new = ct.assign(g, scale_init)
        b_new = ct.assign(b, -m_init*scale_init)

        x_init = ct.reshape(scale_init, (num_filters, 1, 1))*(x_init-ct.reshape(m_init, (num_filters, 1, 1))) + ct.reshape(g_new + b_new, (num_filters, 1, 1))*0
        if nonlinearity is not None:
            x_init = nonlinearity(x_init)
        return x_init
        
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = l2_normalize(V, axes=(1, 2, 3))        
        W = ct.reshape(g, (num_filters, 1, 1, 1)) * V_norm

        x = ct.convolution(W, x, strides=x_channels_shape + strides, auto_padding=paddings) + ct.reshape(b, (num_filters, 1, 1))

        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def deconv2d(x, num_filters, filter_shape=(3,3), strides=(1,1), pad=True, nonlinearity=None, init=global_init, init_scale=1., counters={}, first_run=False):
    ''' Deconvolution layer. '''

    scope = get_name('deconv2d', counters)    
    output_channels_shape = _as_tuple(num_filters)
    x_shape               = x.shape # CHW
    x_channels_shape      = _as_tuple(x.shape[0])
    paddings              = (False,) + (pad,)*len(filter_shape)

    if pad:
        output_shape = (num_filters, x_shape[1] * strides[0], x_shape[2] * strides[1])
    else:
        output_shape = (num_filters, x_shape[1] * strides[0] + filter_shape[0] - 1, x_shape[2] * strides[1] + filter_shape[1] - 1)

    if first_run:
        V = ct.parameter(x_channels_shape + output_channels_shape + filter_shape, init=init, name='V'); set_parameter(scope, 'V', V)
        g = ct.parameter(output_channels_shape, init=global_g_init, name='g'); set_parameter(scope, 'g', g)
        b = ct.parameter(output_channels_shape, name='b'); set_parameter(scope, 'b', b)

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = l2_normalize(V, axes=(0, 2, 3))
        x_init = ct.convolution_transpose(V_norm, x, strides=x_channels_shape + strides, output_shape=output_shape, auto_padding=paddings)

        m_init, v_init = moments(x_init, axes=(ct.Axis.default_batch_axis(),1,2))
        scale_init = init_scale / ct.sqrt(v_init + 1e-8)
        g_new = ct.assign(g, scale_init)
        b_new = ct.assign(b, -m_init*scale_init)

        x_init = ct.reshape(scale_init, (num_filters, 1, 1))*(x_init-ct.reshape(m_init, (num_filters, 1, 1))) + ct.reshape(g_new + b_new, (num_filters, 1, 1))*0
        if nonlinearity is not None:
            x_init = nonlinearity(x_init)
        return x_init
        
    else:
        V,g,b = get_parameters(scope, ['V','g','b'])

        # use weight normalization (Salimans & Kingma, 2016)
        V_norm = l2_normalize(V, axes=(0, 2, 3))
        W = ct.reshape(g, (1, num_filters, 1, 1)) * V_norm

        x = ct.convolution_transpose(W, x, strides=x_channels_shape + strides, output_shape=output_shape, auto_padding=paddings) + ct.reshape(b, (num_filters, 1, 1))
        
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s  = x.shape
    x = ct.reshape(x, (s[0], np.prod(s[1:])))
    x = dense(x, num_units, **kwargs)
    return ct.reshape(x, (num_units,)+s[1:])

def masked_resnet(x, h=None, init=ct.glorot_uniform()):
    '''
    Residual block, from PixelRNN paper (https://arxiv.org/pdf/1601.06759v3.pdf), used
    to build up the hidden layers in PixelCNN.
    '''
    num_filters_2h = x.shape[0]
    num_filters_h  = num_filters_2h // 2

    l = masked_conv2d(x, num_filters_h, filter_shape=(1,1), mask_type='b', h=h, init=init)
    l = masked_conv2d(l, num_filters_h, filter_shape=(3,3), mask_type='b', h=h, init=init)
    l = masked_conv2d(l, num_filters_2h, filter_shape=(1,1), mask_type='b', h=h, init=init)

    return x + l

def masked_gated_resnet(input_v, input_h, filter_shape, h=None, init=ct.glorot_uniform()):
    '''
    Condition PixelCNN building block (https://arxiv.org/pdf/1606.05328v2.pdf), it is composed of vertical stack
    and horizontal stack with resnet connection. 
    '''
    num_filters = input_h.shape[0]
    v = masked_conv2d(input_v, 2*num_filters, filter_shape=filter_shape, strides=(1,1), mask_type='b', h=h, bias=False, init=init)
    h = masked_conv2d(input_h, 2*num_filters, filter_shape=(1, filter_shape[1]), strides=(1,1), bias=False, init=init)    
    h = h + masked_conv2d(v, 2*num_filters, filter_shape=(1,1), strides=(1,1), bias=False, init=init)

    # Vertical stack
    v = ct.element_times(ct.tanh(v[:num_filters,:,:]), ct.sigmoid(v[num_filters:2*num_filters,:,:]))

    # Horizontal stack
    h = ct.element_times(ct.tanh(h[:num_filters,:,:]), ct.sigmoid(h[num_filters:2*num_filters,:,:]))
    h = masked_conv2d(h, num_filters, filter_shape=(1,1), strides=(1,1), bias=False, init=init)
    h = h + input_h

    return v, h

def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=global_init, dropout_p=0., counters={}, first_run=False):
    xs = x.shape
    num_filters = xs[0]

    c1 = conv(nonlinearity(x), num_filters, counters=counters, first_run=first_run)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters, counters=counters, first_run=first_run)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = ct.dropout(c1, dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1, counters=counters, first_run=first_run)

    # add projection of h vector if included: conditional generation
    if h is not None:
       Wh = ct.parameter(h.shape + (2 * num_filters,), init=init, name='Wh')
       c2 = c2 + ct.reshape(ct.times(h, Wc), (2 * num_filters, 1, 1))

    a = c2[:num_filters,:,:]
    b = c2[num_filters:2*num_filters,:,:]
    c3 = a * ct.sigmoid(b)
    return x + c3

''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = x.shape
    # return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1) BHWC
    return ct.splice(zeros((xs[0], 1, xs[2])), x[:, :xs[1]-1, :], axis=1)             # CHW

def right_shift(x):
    xs = x.shape
    # return f.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)  BHWC
    return ct.splice(zeros((xs[0], xs[1], 1)), x[:, :, :xs[2]-1], axis=2)             # CHW

def down_shifted_conv2d(x, num_filters, filter_shape=(2,3), strides=(1,1), **kwargs):
    # x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    xs = x.shape
    if filter_shape[0] > 1:
        x = ct.splice(zeros((xs[0], filter_shape[0] - 1, xs[2])), x, axis=1)
        xs = x.shape
    x = ct.splice(zeros((xs[0], xs[1], int((filter_shape[1]-1)/2))), x, axis=2)
    xs = x.shape
    x = ct.splice(x, zeros((xs[0], xs[1], int((filter_shape[1]-1)/2))), axis=2)
    x = conv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    return x

def down_shifted_deconv2d(x, num_filters, filter_shape=(2,3), strides=(1,1), **kwargs):
    x = deconv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    xs = x.shape
    #      x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]   BHWC
    return x[:, :(xs[1] - filter_shape[0] + 1), int((filter_shape[1] - 1) / 2):(xs[2] - int((filter_shape[1] - 1) / 2))] # CHW

def down_right_shifted_conv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    # x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]])
    xs = x.shape
    if filter_shape[0] > 1:
        x = ct.splice(zeros((xs[0], filter_shape[0] - 1, xs[2])), x, axis=1)
        xs = x.shape
    if filter_shape[1] > 1:
        x = ct.splice(zeros((xs[0], xs[1], filter_shape[1] - 1)), x, axis=2)
    return conv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)

def down_right_shifted_deconv2d(x, num_filters, filter_shape=(2,2), strides=(1,1), **kwargs):
    x = deconv2d(x, num_filters, filter_shape=filter_shape, pad=False, strides=strides, **kwargs)
    xs = x.shape
    #      x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]  BHWC
    return x[:, :(xs[1] - filter_shape[0] + 1):, :(xs[2] - filter_shape[1] + 1)] # CHW
