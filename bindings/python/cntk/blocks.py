# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# blocks -- basic building blocks that are semantically not layers (not used in a layered fashion)
#           e.g. the LSTM
# TODO: This has become too large. Need to break out some locally used stuff into another module.

# TODO: further clean up the dependencies
from __future__ import division
import numpy as np
from cntk import parameter, constant, input_variable, placeholder_variable, combine, alias
from cntk.axis import Axis
from cntk.ops import times, slice, sigmoid, tanh, log, exp, past_value, future_value
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, _as_tuple
from cntk.initializer import glorot_uniform
from _cntk_py import InferredDimension
from cntk.default_options import *

# TODO: As you are on the level of cntk here, you could use relative imports:
# from .ops.functions import Function
# No -> SystemError: Parent module '' not loaded, cannot perform relative import
from cntk.ops.functions import Function
from cntk.ops.variables import Variable

_trace_layers = False
#_trace_layers = True  # uncomment this to log creation of graph through layers

_INFERRED = (InferredDimension,)  # as a tuple, makes life easier

# call this for all untested branches
def UntestedBranchError(name):
    raise NotImplementedError("Untested code branch: " + name)

# resolve activation option against current default
#def _resolve_activation(activation):
#    # if none is passed to caller then use the default
#    if activation is _default_sentinel:
#        activation = _current_default_options.activation
#    # activation=None is implemented as activation=identity
#    if activation is None:
#        activation = identity
#    return activation

# create the complete initializer for a given 'init' parameter, to pass to parameter()
# This is called from Parameter() and every place that injects rank parameters.
# It does a few things:
#  - maps init_default_override_or_glorot_uniform to default  --TODO: we should have a global setting for that
#  - creates a new initializer object from an existing one, while updating members
# TODO: remove default resolution, only make this a conversion; then rename
def _initializer_for(init, rank_params=None):
    if init is None:
        raise ValueError("init parameter cannot be None")

    # if default then select
    #if init is _default_sentinel_init:
    #    init = _current_default_options.init
    #elif init is _default_sentinel_init_bias:
    #    init = _current_default_options.init_bias

    # scalar constant: that's it, nothing further to do here
    if np.isscalar(init):
        # BUGBUG: this is sometimes required when dimensions are unknown; shouldn't.
        from _cntk_py import constant_initializer
        return constant_initializer(init)
        #return init # TODO: change to this once this works, e.g. for layers.BatchNormalization()

    # implant additional rank parameters
    if rank_params:
        from cntk.initializer import initializer_with_rank
        init = initializer_with_rank(init, **rank_params)

    return init

# create a type specifier; that is, all arguments to instantiate a Placeholder or Input
# All are optional, meaning unspecified.
# TODO: move to class Value?
def Type(shape=None, dtype=None, is_sparse=None, dynamic_axes=None):
    r = dict()
    if shape is not None:
        r['shape'] = shape
    if dtype is not None:
        r['dtype'] = dtype
    if is_sparse is not None:
        r['is_sparse'] = is_sparse
    if dynamic_axes is not None:
        r['dynamic_axes'] = dynamic_axes
    return Record(**r)

# turn a Function into a Block, with a new name and an optional dictionary of named parameters
# If passed function is an actual Python function, it will be executed with Placeholders as inputs.
# All layers functions call this at the end.
# BUGBUG: does not actually exist yet, faking it
# BUGBUG: should create a new object, but does it in-place instead. Works for current usage, but should be fixed.
# BUGBUG: using combine causes an error ater, so the name actually does not get changed
# BUGBUG: combine like this won't work for functions with multiple outputs (LSTM)
def Block(f, op_name, members={}):

    #from inspect import isfunction
    #if isfunction(f):
    #    f = Function(f, members)

    #f = combine([f], op_name)  # 'combine' to create a separate identity so we can reassign the debug name --BUGBUG: "Unknown DataType"
    #_name_node(f, op_name) ; _extend_Function(f)  # debugging

    #p = f.placeholders
    #a = f.arguments
    #f = alias(f, name=op_name)
    #p1 = f.placeholders
    #a1 = f.arguments
    # BUGBUG: alias() does not work. Once I add this, some clone() call no longer finds the placeholder
    #if len(p) != len(p1) or len(a) != len(a1):
    #    raise AssertionError('')

    for key in members:   # self.__dict__.update(args_dict)
        f.__dict__[key] = members[key]
    return f

# TODO: Move this into the lower layer where these are defined.
# some mappings--these currently exist only so that I can name the nodes for debugging
def Parameter(shape, init, dtype=default_override_or(np.float32), name=''):
    dtype = get_default_override(Parameter, dtype=dtype)
    init = _initializer_for(init)
    p = parameter(shape, init=init, dtype=dtype, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'parameter')   # these are factory methods for things with state

def Constant(init, shape=None, dtype=default_override_or(np.float32), name=''):
    dtype = get_default_override(Constant, dtype=dtype)
    p = constant(init, shape=shape, dtype=dtype, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'constant')   # these are factory methods for things with state

# TODO: this function should not be necessary anymore
def Input(shape, dtype=default_override_or(np.float32), needs_gradient=True, is_sparse=False,
          dynamic_axes=Axis.default_input_variable_dynamic_axes(), name=''):
    dtype = get_default_override(Input, dtype=dtype)
    return _name_node(input_variable(shape=shape, dtype=dtype, needs_gradient=needs_gradient, is_sparse=is_sparse,
                                     dynamic_axes=dynamic_axes, name=name), 'input')

def Placeholder(shape=None, dynamic_axes=None, is_sparse=False, name='placeholder'):
    #p = placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, is_sparse=is_sparse, name=name) # TODO: use (*args, **kwargs)?
    # BUGBUG: placeholder does not know is_sparse
    p = placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, name=name) # TODO: use (*args, **kwargs)?
    _name_node(p, name)
    if _trace_layers:
        print("new " + _node_description(p))
    return p

# there is only one identity function
# TODO: This should become a C++-side Function, e.g. like sigmoid
@Function
def identity(keep):
    return combine([keep])

# This takes enable_self_stabilization as a flag that allows to disable itself. Useful if this is a global default.
def Stabilizer(steepness=4, enable_self_stabilization=default_override_or(True)):

    enable_self_stabilization = get_default_override(Stabilizer, enable_self_stabilization=enable_self_stabilization)

    if not enable_self_stabilization: # disabled (typically through global option; otherwise one would not call this in the first place)
        return identity

    # parameters bound to this Function
    param = Parameter((1), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    # TODO: compute this strange value directly in Python
    beta = log (1 + exp (steepness * param)) * (1 / steepness)   # perf BUGBUG: "log() / steepness" should optimize to the samething   --TODO: change in Python

    # expression
    @Function
    def stabilize(x):
        # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
        # this behaves linear for weights around 1, yet guarantees positiveness
        # TODO: risk of confusion; can these functions be namespaced?
        return beta * x
    return Block(stabilize, 'Stabilizer', Record(beta=beta))

# recurrent block of type 'LSTM', 'GRU', or RNNUnit
def _RecurrentBlock(type, shape, cell_shape, activation, use_peepholes,
                    init, init_bias,
                    enable_self_stabilization):

    has_projection = cell_shape is not None
    has_aux = False

    if has_aux:
        UntestedBranchError("LSTM, has_aux option")

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape
    if len(shape) != 1 or len(cell_shape) != 1:
        raise ValueError("LSTM: shape and cell_shape must be vectors (rank-1 tensors)")
        # otherwise we'd need to fix slicing and Param initializers

    stack_axis = -1  # stacking along the fastest-changing one, to match BS
    # determine stacking dimensions
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[stack_axis]
    cell_shape_list[stack_axis] = stacked_dim * {
        'RNNUnit': 1,
        'GRU': 3,
        'LSTM': 4
    }[type]
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * {
        'RNNUnit': 1,
        'GRU': 2,
        'LSTM': 4
    }[type]
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked,   init=init_bias, name='b')                              # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='W')                              # input
    A  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='A') if has_aux else None         # aux input (optional)  --TODO: remove
    H  = Parameter(shape     + cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden
    H1 = Parameter(shape     + cell_shape,           init=init,      name='H') if type == 'GRU' else None   # hidden-to-hidden
    Ci = Parameter(            cell_shape,           init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,           init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,           init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init) if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer() if enable_self_stabilization else identity
    Sdc = Stabilizer() if enable_self_stabilization else identity
    Sct = Stabilizer() if enable_self_stabilization else identity
    Sht = Stabilizer() if enable_self_stabilization else identity

    #from collections import OrderedDict

    # define the model function itself
    # general interface for Recurrence():
    #   (x, all previous outputs delayed) --> (outputs and state)
    # where
    #  - the first output is the main output, e.g. 'h' for LSTM
    #  - the remaining outputs, if any, are additional state
    #  - if for some reason output != state, then output is still fed back and should just be ignored by the recurrent block

    # LSTM model function
    # in this case:
    #   (x, dh, dc) --> (h, c)
    @Function
    def lstm(x, dh, dc):

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dhs, H) + times(aux, A) if has_aux else \
                b + times(x, W) + times(dhs, H)

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation (bit_proj)              # applied to tanh of input network

        ft = sigmoid (peep (ft_proj, dcs, Cf))        # forget-me-not gate(t)
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
        ht = ot * activation (ct)                     # applied to tanh(cell(t))

        c = ct                                        # cell value
        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        return (Function.NamedOutput(h=h), Function.NamedOutput(c=c))
        #return OrderedDict([('h', h), ('c', c)])

    # GRU model function
    # in this case:
    #   (x, dh) --> (h)
    # e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    # TODO: Is this the same definition as NVidia's? Should we enable multiple definitions of this?
    # BUGBUG: gru(x,dh,dc) passes, too. Since 'dc' is not referenced, it is just ignored. Also when routing it through combine().
    @Function
    def gru(x, dh):

        dhs = Sdh(dh)  # previous value, stabilized
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        projx3 = b + times(x, W) + times(aux, A) if has_aux else \
                 b + times(x, W)
        projh2  = times(dhs, H)

        zt_proj = slice (projx3, stack_axis, 0*stacked_dim, 1*stacked_dim) + slice (projh2, stack_axis, 0*stacked_dim, 1*stacked_dim)
        rt_proj = slice (projx3, stack_axis, 1*stacked_dim, 2*stacked_dim) + slice (projh2, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ct_proj = slice (projx3, stack_axis, 2*stacked_dim, 3*stacked_dim)

        zt = sigmoid (zt_proj)        # update gate z(t)

        rt = sigmoid (rt_proj)        # reset gate r(t)

        rs = dhs * rt        # "cell" c
        ct = activation (ct_proj + times(rs, H1))

        ht = (1 - zt) * ct + zt * dhs # hidden state ht / output

        # for comparison: CUDNN_GRU
        # i(t) = sigmoid(W_i x(t) +          R_i h(t-1)  + b_Wi + b_Ru)
        # r(t) = sigmoid(W_r x(t) +          R_r h(t-1)  + b_Wr + b_Rr)   --same up to here
        # h'(t) =   tanh(W_h x(t) + r(t) .* (R_h h(t-1)) + b_Wh + b_Rh)   --r applied after projection? Would make life easier!
        # h(t) = (1 - i(t) .* h'(t)) + i(t) .* h(t-1)                     --wrong bracketing??

        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        return Function.NamedOutput(h=h)

    @Function
    def rnn(x, dh):
        dhs = Sdh(dh)  # previous value, stabilized
        ht = activation (times(x, W) + times(dhs, H) + b)
        h = times(Sht(ht), Wmr) if has_projection else \
            ht
        return Function.NamedOutput(h=h)

    # return the corresponding lambda as a CNTK Function
    function = Block({
        'RNNUnit': rnn,
        'GRU':     gru,
        'LSTM':    lstm
    }[type], type)

    # we already know our state input's shapes, so implant the shape there
    # This is part of the contract with Recurrence(), which relies on this.
    # BUGBUG: If V2 type inference could handle unknown shapes here, we would not need this.
    # BUGBUG: This fails if recurrent cells are composed, since we won't know the true feedback dimension. V2 inference should just handle this.
    function.replace_placeholders({ function.placeholders[index] : Placeholder(shape=shape) for index, shape in {
        'RNNUnit': { 1: shape },
        'GRU':     { 1: shape },
        'LSTM':    { 1: shape, 2: cell_shape }
    }[type].items() })

    return function

# LSTM block
# returns a function (input, prev_h, prev_c -> h, c)
def LSTM(shape, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False)):

    activation                = get_default_override(RNNUnit, activation=activation)
    use_peepholes             = get_default_override(LSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(LSTM, init=init)
    init_bias                 = get_default_override(LSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(LSTM, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('LSTM', shape, cell_shape, activation=activation, use_peepholes=use_peepholes,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization)

# RNN block
# returns a function (input, prev_h) -> h)
#   h = activation (W * input + R * prev_h + b)
# TODO: needs better name
def RNNUnit(shape, cell_shape=None, activation=default_override_or(sigmoid),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False)): # (x, prev_h) -> (h)

    activation                = get_default_override(RNNUnit, activation=activation)
    init                      = get_default_override(RNNUnit, init=init)
    init_bias                 = get_default_override(RNNUnit, init_bias=init_bias)
    enable_self_stabilization = get_default_override(RNNUnit, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('RNNUnit', shape, cell_shape, activation=activation, use_peepholes=False,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization)

# GRU block
def GRU(shape, cell_shape=None, activation=default_override_or(tanh),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False)): # (x, prev_h) -> (h)

    activation                = get_default_override(GRU, activation=activation)
    init                      = get_default_override(GRU, init=init)
    init_bias                 = get_default_override(GRU, init_bias=init_bias)
    enable_self_stabilization = get_default_override(GRU, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('GRU', shape, cell_shape, activation=activation, use_peepholes=False,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization)

# TODO:
#  - get last() to work

# TODO in C++ code after next update:
#  - seq2seq: support initial_state with batch dimension, as a V2-V1 compilation step
#  - elementwise: sequence broadcasting for all elementwise operations, as a V2-V1 compilation step
