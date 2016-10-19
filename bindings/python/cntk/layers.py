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
from cntk.learner import sgd, fsadagrad
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
import itertools
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, _as_tuple
from cntk.blocks import *
from cntk.blocks import _name_and_extend_Function, _wrap_rename_Function, _trace_layers  # (debugging)
from cntk.initializer import glorot_uniform

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
# TODO: move these out from examples
#from examples.common.nn import slice, sigmoid, log, tanh, past_value, future_value, print_training_progress, negate

from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# this is what we initialize weight matrices from by default
from cntk.blocks import _default_initializer

# Linear -- create a fully-connected linear projection layer
# Note: shape may describe a tensor as well.
# TODO: change to new random-init descriptor
# inputRank given: number of zeroes to add to W (mapRank must not be given)
# mapRank   given: expand W to leave exactly mapRank axes (inputRank must not be given)
# none      given: expand W to all (same as mapRank=0)
def Linear(shape, _inf, bias=True, init=_default_initializer, init_bias=0, input_rank=None, map_rank=None):
    out_shape = _as_tuple(shape)

    # TODO: implement the full semantics of the BrainScript code
    #inputShape =
    #    if       BS.Constants.IsNone (inputRank) then Inferred  # not given: one Inferred, which will get expanded
    #    else if !BS.Constants.IsNone (mapRank)   then Fail ("'inputRank' and 'mapRank' cannot be specified at the same time.")
    #    else Repeat (inputRank, Inferred)
    #W = ParameterTensor {_ConcatArrays (outDim, inputShape), init=init, initValueScale=initValueScale}
    #b = ParameterTensor {outDim, initValue=0}
    #outputRank = Length (_AsArray (outDim)) # support outputs with tensor layouts
    #inferInputRankToMap =
    #    if      !BS.Constants.IsNone (inputRank) then -1  # means not specified
    #    else if  BS.Constants.IsNone (mapRank)   then 0   # default to 'use all input dims'
    #    else mapRank
    #apply (x) =
    #    if bias
    #    then Times (W, x, outputRank=outputRank, inferInputRankToMap=inferInputRankToMap) + b
    #    else Times (W, x, outputRank=outputRank, inferInputRankToMap=inferInputRankToMap)

    W = Parameter(_inf.shape + out_shape, init=init     , name='W')
    b = Parameter(             out_shape, init=init_bias, name='b') if bias else None
    x = Placeholder(_inf=_inf, name='linear_arg')
    apply_x = Function.__matmul__(x, W) + b if bias else \
              Function.__matmul__(x, W)
    _name_and_extend_Function(apply_x, 'Linear')
    return apply_x
    # TODO: how to break after the else?

def Dense(shape, _inf, bias=True, init=_default_initializer, init_bias=0, input_rank=None, map_rank=None, activation=None):
    if activation is None:  # TODO: change default to identity once we no longer need _inf
        activation = Identity(_inf=shape)
    apply_x = Linear(shape, _inf, bias=bias, init=init, init_bias=init_bias, input_rank=input_rank, map_rank=map_rank) \
           >> activation
    # TODO: Any way to do some similar pattern ^^ without backslash?
    _name_and_extend_Function(apply_x, 'Dense')
    return apply_x

# Embedding -- create a linear embedding layer
# TODO: after removing loading from file, now we have two similar params, weight and init which seems redundant.
# TODO: Once _inf is gone, change interface to pass weights as a Constant, e.g.
#       Embedding(shape, constant(np.load('PATH')))
#       Not nice since now we don't need the output shape either. Grmpf.
def Embedding(shape, _inf, weights=None, init=_default_initializer, transpose=False):
    shape = _as_tuple(shape)
    full_shape = (shape + _inf.shape) if transpose else (_inf.shape + shape)
    if weights is None:  # no weights given: learn the embedding
        E = Parameter(full_shape, init=init, name='E')
    else:                # weights given: use them as constant
        UntestedBranchError("Embedding, from constant")
        E = Constant(full_shape, init=weights, name='E')  # TODO: can 'weights' be a CNTK object already? Then how to do this?
    x = Placeholder(_inf=_inf, name='embedding_arg')
    apply_x = Function.__matmul__(E, x) if transpose else \
              Function.__matmul__(x, E)     # x is expected to be sparse one-hot
    _name_and_extend_Function(apply_x, 'Embedding')
    return apply_x

def Stabilizer(_inf, steepness=4):
    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness

    # parameters
    param = Parameter((1), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    # TODO: compute this strange value directly in Python

    # application
    x = Placeholder(_inf=_inf, name='stabilizer_arg')
    # TODO: risk of confusion; can these functions be namespaced?
    beta = log (1 + exp (steepness * param)) / steepness
    apply_x = beta * x
    _name_and_extend_Function(apply_x, 'Stabilizer')
    return apply_x

def Recurrence(over, _inf=None, go_backwards=False, initial_state=None):
    # helper to compute previous value
    # can take a single Variable/Function or a tuple
    if go_backwards:
        UntestedBranchError("Recurrence, go_backwards option")
    def previous_hook(state):
        if hasattr(state, 'outputs'):
           outputs = state.outputs
           if len(outputs) > 1:  # if multiple then apply to each element
               return tuple([previous_hook(s) for s in outputs])
        # not a tuple: must be a 'scalar', i.e. a single element
        return past_value  (state, initial_state) if not go_backwards else \
               future_value(state, initial_state)
    x = Placeholder(_inf=_inf, name='recurrence_arg')
    prev_state_forward = over.create_placeholder() # create a placeholder or a tuple of placeholders
    f_x_h_c = over(x, prev_state_forward) # apply the recurrent over
    # this returns a Function (x, (h_prev, c_prev)) -> (h, c)
    h = f_x_h_c.outputs[0]  # 'h' is a Variable (the output of a Function that computed it)
    if _trace_layers:
        _log_node(h)
        _log_node(combine([h.owner]))
    prev_state = previous_hook(f_x_h_c)  # delay (h, c)
    replacements = { value_forward: value.output for (value_forward, value) in zip(list(prev_state_forward), list(prev_state)) }
    f_x_h_c.replace_placeholders(replacements)  # binds _h_c := prev_state
    apply_x = combine([h.owner])     # the Function that yielded 'h', so we get to know its inputs
    # apply_x is a Function x -> h
    _name_and_extend_Function(apply_x, 'Recurrence')
    if _trace_layers:
        _log_node(apply_x)
    return apply_x
