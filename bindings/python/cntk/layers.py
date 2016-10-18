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
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, _as_tuple
from cntk.blocks import *  # TODO: reduce to what we actually use
from cntk.blocks import _extend_Function, _name_and_extend_Function, _wrap_rename_Function, _trace_layers  # (debugging)
from cntk.initializer import glorot_uniform

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
# TODO: move these out from examples
#from examples.common.nn import slice, sigmoid, log, tanh, past_value, future_value, print_training_progress, negate

from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# this is what we initialize weight matrices from by default
from cntk.blocks import _default_initializer, _Inferred

# Dense -- create a fully-connected linear projection layer with optional non-linear activation
# Note: shape may describe a tensor as well.
# input_rank given: number of inferred axes to add to W (map_rank must not be given)
# map_rank   given: expand W to leave exactly mapRank axes (input_rank must not be given)
# none       given: expand W to all (same as map_rank=0)
def Dense(shape, init=_default_initializer, activation=identity, input_rank=None, map_rank=None, bias=True, init_bias=0):
    output_shape = _as_tuple(shape)

    if input_rank is not None and map_rank is not None:
        raise ValueError("Dense: input_rank and map_rank cannot be specified at the same time.")

    # determine meaning of axes
    # W gets dimension (input_shape + shape)
    # where input_shape is determined as:
    #  - by default, equal to the dimensions of the input passed to Dense()
    #  - if input_rank is given, then the last 'input_rank' dimensions of the input
    #  - if map_rank is given, then the first 'map_rank' dimensions of the input
    # where input_rank and map_rank are mutuallly exclusive.
    output_rank = len(output_shape)   # support outputs with tensor layouts

    # If input_rank not given then pass a single _Inferred; map_rank if given will determine the input_rank.
    # The dimension inference may still create multiple axes.
    input_shape = _Inferred * (input_rank if input_rank is not None else 1)

    if input_rank is not None:
        UntestedBranchError("Dense, input_rank option not implemented")
        infer_input_rank_to_map = -1 # means map_rank is not specified; input_rank rules
    elif map_rank is None:
        infer_input_rank_to_map = 0  # neither given: default to 'infer W to use all input dims'
    else:
        UntestedBranchError("Dense, map_rank option not implemented")
        infer_input_rank_to_map = map_rank  # infer W to use all input dims except the last 'map_rank' ones

    # parameters bound to this Function
    W = Parameter(input_shape + output_shape, init=init     , name='W')
    b = Parameter(              output_shape, init=init_bias, name='b') if bias else None

    # expression of this function
    x = Placeholder(name='dense_arg')
    apply_x = times(x, W, output_rank=output_rank, infer_input_rank_to_map=infer_input_rank_to_map)
    if b:
        apply_x = apply_x + b
    _extend_Function(apply_x)  # (this gets us the >> operator  --TODO: remove once Function natively supports this)
    apply_x = apply_x >> activation
    _name_and_extend_Function(apply_x, 'Dense')
    return apply_x

# Embedding -- create a linear embedding layer
# To create an embedding from a file, use this:
#   Embedding(shape, Constant(np.load('PATH')))
# TODO: remove shape in case of Constant
def Embedding(shape, init=_default_initializer, transpose=False):
    shape = _as_tuple(shape)
    weights = None   # TODO: finish the Constant() thing
    if weights is None:  # no weights given: learn the embedding
        full_shape = _Inferred + shape
        E = Parameter(full_shape, init=init, name='E')
    else:                # weights given: use them as constant
        UntestedBranchError("Embedding, from constant")
        # TODO: infer full_shape from weights? Which in turn should be a constant... lots of TODO here
        full_shape = (shape + _Inferred) if transpose else (_Inferred + shape)
        E = Constant(full_shape, init=weights, name='E')  # TODO: can 'weights' be a CNTK object already? Then how to do this?
    x = Placeholder(name='embedding_arg')
    apply_x = Function.__matmul__(E, x) if transpose else \
              Function.__matmul__(x, E)     # x is expected to be sparse one-hot
    _name_and_extend_Function(apply_x, 'Embedding')
    return apply_x

def Recurrence(over, _inf=None, go_backwards=False, initial_state=None):
    # helper to compute previous value
    # can take a single Variable/Function or a tuple
    if go_backwards:
        UntestedBranchError("Recurrence, go_backwards option")
    def previous_hook(state):
        if hasattr(state, 'outputs'):
           outputs = state.outputs()
           if len(outputs) > 1:  # if multiple then apply to each element
               return tuple([previous_hook(s) for s in outputs])
        # not a tuple: must be a 'scalar', i.e. a single element
        return past_value  (state, initial_state) if not go_backwards else \
               future_value(state, initial_state)
    x = Placeholder(_inf=_inf, name='recurrence_arg')
    #x = Placeholder(name='recurrence_arg') # BUGBUG: Fails with "Variable with unknown dynamic axes detected when compiling the Function graph!"
    prev_state_forward = over.create_placeholder() # create a placeholder or a tuple of placeholders
    f_x_h_c = over(x, prev_state_forward) # apply the recurrent over
    # this returns a Function (x, (h_prev, c_prev)) -> (h, c)
    h = f_x_h_c.outputs()[0]  # 'h' is a Variable (the output of a Function that computed it)
    if _trace_layers:
        _log_node(h)
        _log_node(combine([h.owner()]))
    prev_state = previous_hook(f_x_h_c)  # delay (h, c)
    replacements = { value_forward: value.output() for (value_forward, value) in zip(list(prev_state_forward), list(prev_state)) }
    f_x_h_c.replace_placeholders(replacements)  # binds _h_c := prev_state
    apply_x = combine([h.owner()])     # the Function that yielded 'h', so we get to know its inputs
    # apply_x is a Function x -> h
    _name_and_extend_Function(apply_x, 'Recurrence')
    if _trace_layers:
        _log_node(apply_x)
    return apply_x
