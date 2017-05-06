import sys
import os

import numpy as np
import cntk as ct
from . import nn as nn

def sample_from(l, nr_mix, loss='mixture'):
    if loss == 'category':
        return np_softmax_256_sample(l)
    elif loss == 'mixture':
        return np_sample_from_discretized_mix_logistic(l, nr_mix)
    return None

def np_sample_from_discretized_mix_logistic(l, nr_mix=10):
    ls = l.shape # NCHW
    xs = (ls[0], 3) + ls[2:]

    # unpack parameters
    logit_probs = l[:,:nr_mix,:,:]
    l = np.reshape(l[:,nr_mix:,:,:], (xs[0], nr_mix*3)+xs[1:])

    # sample mixture indicator from softmax
    sel = nn.one_hot(np.argmax(logit_probs - np.log(-np.log(np.random.uniform(1e-5, 1. - 1e-5, logit_probs.shape).astype('f'))), axis=1), depth=nr_mix, axis=1 , dtype=np.float32)
    sel = np.reshape(sel, (xs[0], nr_mix, 1) + xs[2:])

    # select logistic parameters
    means = np.sum(l[:,:nr_mix,:,:,:]*sel,1)
    log_scales = np.maximum(np.sum(l[:,nr_mix:2*nr_mix,:,:,:]*sel,1), -7.)
    coeffs = np.sum(np.tanh(l[:,2*nr_mix:3*nr_mix,:,:,:])*sel,1)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    # u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    u = np.random.uniform(1e-5, 1. - 1e-5, means.shape).astype('f')

    x = means + np.exp(log_scales)*(np.log(u) - np.log(1. - u))
    x0 = np.minimum(np.maximum(x[:,0,:,:], -1.), 1.)
    x1 = np.minimum(np.maximum(x[:,1,:,:] + coeffs[:,0,:,:]*x0, -1.), 1.)
    x2 = np.minimum(np.maximum(x[:,2,:,:] + coeffs[:,1,:,:]*x0 + coeffs[:,2,:,:]*x1, -1.), 1.)

    image = np.concatenate((np.reshape(x0,(xs[0],1)+xs[2:]), 
                            np.reshape(x1,(xs[0],1)+xs[2:]), 
                            np.reshape(x2,(xs[0],1)+xs[2:])), axis=1)
    return (image + 1.)*127.5

def np_sample_from_discretized_mix_logistic_NHWC(l, nr_mix=10):
    l = np.ascontiguousarray(np.transpose(l, (0,2,3,1))) # From NCHW to NHWC
    ls = l.shape
    xs = ls[:-1] + (3,)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = np.reshape(l[:, :, :, nr_mix:], xs + (nr_mix*3,))

    # sample mixture indicator from softmax
    sel = nn.one_hot(np.argmax(logit_probs - np.log(-np.log(np.random.uniform(1e-5, 1. - 1e-5, logit_probs.shape).astype('f'))), axis=3), depth=nr_mix, dtype=np.float32)
    sel = np.reshape(sel, xs[:-1] + (1,nr_mix))    

    # select logistic parameters
    means = np.sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = np.maximum(np.sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = np.sum(np.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = np.random.uniform(1e-5, 1. - 1e-5, means.shape).astype('f')

    x = means + np.exp(log_scales)*(np.log(u) - np.log(1. - u))
    x0 = np.minimum(np.maximum(x[:,:,:,0], -1.), 1.)
    x1 = np.minimum(np.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = np.minimum(np.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)

    # (NHWC)
    image = np.concatenate((np.reshape(x0, xs[:-1]+(1,)), 
                            np.reshape(x1, xs[:-1]+(1,)), 
                            np.reshape(x2, xs[:-1]+(1,))), axis=3)
    image = (image + 1.)*127.5
    return np.ascontiguousarray(np.transpose(image, (0,3,1,2))) # From NHWC to NCHW

def np_softmax_256_sample(l):
    # Based on PixelRNN paper (https://arxiv.org/pdf/1601.06759v3.pdf)
    ls = l.shape

    # l: (B, 3x256, 32, 32) to (B, 256, 3, 32, 32)
    l = np.reshape(l, (ls[0],) + (256,3,32,32))
    sel = nn.one_hot(np.argmax(l - np.log(-np.log(np.random.uniform(1e-5, 1. - 1e-5, l.shape).astype('f'))), axis=1), depth=256, axis=1, dtype=np.float32)
    x = np.sum(l*sel, 1)
    return x
