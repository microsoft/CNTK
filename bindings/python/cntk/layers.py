# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# layers -- blocks in the network that are used layer-like, i.e. layered on top of each other
#           e.g. a fully connected layer with non-linearity

# TODO: clean up the dependencies
import numpy as np
import sys
import os
import time
from cntk import DeviceDescriptor, Trainer, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.learner import sgd, fsadagrad, learning_rates_per_sample, momentums_per_sample
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
import itertools
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _print_node
from utils import Record, _as_tuple
from blocks import *
from blocks import _name_and_extend_Function, _wrap_rename_Function  # (debugging)

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
# TODO: move these out from examples
from examples.common.nn import slice, sigmoid, log, tanh, past_value, future_value, print_training_progress, negate

from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# Linear -- create a fully-connected linear projection layer
# Note: shape may describe a tensor as well.
# TODO: change to new random-init descriptor
def Linear(shape, _inf, bias=True, init='glorot_uniform', init_value_scale=1, input_rank=None, map_rank=None):
    out_shape = _as_tuple(shape)
    W = Parameter(_inf.shape + out_shape, init=init, init_value_scale=init_value_scale)
    b = Parameter(             out_shape, init='zero') if bias else None
    x = Placeholder(_inf=_inf, name='linear_arg')
    apply_x = Function.__matmul__(x, W) + b if bias else \
              Function.__matmul__(x, W)
    _name_and_extend_Function(apply_x, 'Linear')
    return apply_x
    # TODO: how to break after the else?

# Embedding -- create a linear embedding layer
# TODO: replace embedding_path with a numpy array, and pass it as the "init" parameter
def Embedding(shape, _inf, init='glorot_uniform', init_value_scale=1, embedding_path=None, transpose=False):
    shape = _as_tuple(shape)
    full_shape = (shape + _inf.shape) if transpose else (_inf.shape + shape)
    if embedding_path is None:
        # TODO: how to pass all optional args automatically in one go?
        f = Linear(shape, _inf=_inf, init=init, init_value_scale=init_value_scale)
        _wrap_rename_Function(f, 'Embedding')
        return f
    else:
        E = Parameter(full_shape, initFromFilePath=embeddingPath, learningRateMultiplier=0)  # fixed from file
    x = Placeholder(_inf=_inf, name='embedding_arg')
    apply_x = __matmul__(E, x) if transposed else \
              __matmul__(x, E)     # x is expected to be sparse one-hot
    _name_and_extend_Function(apply_x, 'Embedding')
    return apply_x

def Stabilizer(_inf, steepness=4):
    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness

    # parameters
    param = Parameter((1), init_value=0.99537863)  # 1/steepness*ln (e^steepness-1) for steepness==4
    # TODO: compute this strange value directly in Python

    # application
    x = Placeholder(_inf=_inf, name='stabilizer_arg')
    beta = log (1 + exp (steepness * param)) / steepness
    apply_x = beta * x
    _name_and_extend_Function(apply_x, 'Stabilizer')
    return apply_x

def Recurrence(over, _inf=None, go_backwards=False):
    # helper to compute previous value
    # can take a single Variable/Function or a tuple
    def previous_hook(state):
        if hasattr(state, 'outputs'):
           outputs = state.outputs()
           if len(outputs) > 1:  # if multiple then apply to each element
               return tuple([previous_hook(s) for s in outputs])
        # not a tuple: must be a 'scalar', i.e. a single element
        return past_value(state) if not go_backwards else \
               future_value(state)
    x = Placeholder(_inf=_inf, name='recurrence_arg')
    prev_state_forward = over.create_placeholder() # create a placeholder or a tuple of placeholders
    f_x_h_c = over(x, prev_state_forward) # apply the recurrent over
    # this returns a Function (x, (h_prev, c_prev)) -> (h, c)
    h = f_x_h_c.outputs()[0]  # 'h' is a Variable (the output of a Function that computed it)
    _print_node(h)
    _print_node(combine([h.owner()]))
    prev_state = previous_hook(f_x_h_c)  # delay (h, c)
    replacements = { value_forward: value.output() for (value_forward, value) in zip(list(prev_state_forward), list(prev_state)) }
    f_x_h_c.replace_placeholders(replacements)  # binds _h_c := prev_state
    apply_x = combine([h.owner()])     # the Function that yielded 'h', so we get to know its inputs
    # apply_x is a Function x -> h
    _name_and_extend_Function(apply_x, 'Recurrence')
    _print_node(apply_x)
    return apply_x
