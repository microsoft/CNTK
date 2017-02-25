import sys
import os

import numpy as np
import cntk as ct
from . import nn as nn


def sample_from_discretized_mix_logistic(l,nr_mix):
    l = ct.transpose(ct.transpose(l,0,1),1,2) # From CHW to HWC
    ls = l.shape
    xs = ls[:-1] + (3,)

    # unpack parameters
    logit_probs = l[:, :, :nr_mix]
    l = ct.reshape(l[:, :, nr_mix:], xs + (nr_mix*3,))

    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)



    sel = ct.reshape(sel, xs[:-1] + (1,nr_mix))
    # select logistic parameters
    means = ct.reduce_sum(l[:,:,:,:nr_mix]*sel,4)
    log_scales = nn.maximum(ct.reduce_sum(l[:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = ct.reduce_sum(ct.tanh(l[:,:,:,2*nr_mix:3*nr_mix])*sel,4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)


    x = means + ct.exp(log_scales)*(ct.log(u) - ct.log(1. - u))
    x0 = nn.minimum(nn.maximum(x[:,:,0], -1.), 1.)
    x1 = nn.minimum(nn.maximum(x[:,:,1] + coeffs[:,:,0]*x0, -1.), 1.)
    x2 = nn.minimum(nn.maximum(x[:,:,2] + coeffs[:,:,1]*x0 + coeffs[:,:,2]*x1, -1.), 1.)

    image = ct.splice(ct.reshape(x0,xs[:-1]+(1,)), ct.reshape(x1,xs[:-1]+(1,)), ct.reshape(x2,xs[:-1]+(1,)), axis=3)

    return ct.reshape(image, (3,32,32))
