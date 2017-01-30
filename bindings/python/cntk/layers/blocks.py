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
from cntk import parameter, constant, input_variable, placeholder_variable, combine, alias, sequence
from cntk.axis import Axis
from cntk.ops import times, slice, sigmoid, tanh, log, exp, past_value, future_value
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, RecordWith, _as_tuple
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

# helper to get the initial_state or the default
def _get_initial_state_or_default(initial_state):
    # if initial_state is a tuple (multiple state vars), then apply this recursively to all
    if isinstance(initial_state, tuple):
        return tuple(_get_initial_state_or_default(s) for s in initial_state)
    # if initial state is given and a numeric constant, then turn it into a Constant() object
    elif initial_state is None:
        return Constant(0) # note: don't pass None to past_value, because that would default to float32 --TODO: still the case?
    elif np.isscalar(initial_state):
        return Constant(initial_state, shape=(1))
    else:
        return initial_state # already in good shape: return as is

def _make_tensor_meta(cls_name, **kwargs):
    '''
    BUGBUG: This must be updated
    Variable type descriptor (shape, axes, ...) for use as type annotations in function definitions,
    and as arguments to update_signature().
    Function arguments with such type annotations will compile into a CNTK Function with
    bound Input variables and fully inferred types, e.g. for use with the criterion function.
    Function arguments without such annotation will get bound to placeholders, e.g. for use
    when types are unknowable like for Layers-library functions.
    Example:
        @Function
        def f(x: Tensor[(13,42)]):
            return x * x
    '''
    class TensorMeta(type):
        def __getitem__(self, shape):
            from ..utils import sanitize_shape
            shape = sanitize_shape(shape)
            return Variable.Type(shape, **kwargs) # inject it for @Function 
    return TensorMeta(cls_name, (), {})

# Tensor and SparseTensor contain only a batch axis.
# If you want a sequence, say Seq[Tensor]
# ParameterTensor has no axis. For future use.
Tensor          = _make_tensor_meta('Tensor',       is_sparse=False, dynamic_axes=Axis.default_batch_axis())
SparseTensor    = _make_tensor_meta('SparseTensor', is_sparse=True , dynamic_axes=Axis.default_batch_axis())
#ParameterTensor = _make_tensor_meta('SparseTensor', is_sparse=True , dynamic_axes=[])
tensor = Tensor[-2] # TODO: find the correct symbol for the sentinel value

def _make_seq_meta(cls_name, axes):
    class SeqMeta(type):
        def __getitem__(self, item_type):
            return Variable.Type(**RecordWith(item_type, dynamic_axes=axes))
    return SeqMeta(cls_name, (), {})

Seq = _make_seq_meta('Seq', Axis.default_input_variable_dynamic_axes())
# TODO: accept typing.Sequence instead
# TODO: reject sequences over sequences (for now)

class SequenceOverMeta(type):
    def __getitem__(self, axis):
        return _make_seq_meta('Seq', [Axis.default_batch_axis(), axis])

SequenceOver = SequenceOverMeta('SequenceOver', (), {})

# turn a Function into a Block, with a new name and an optional dictionary of named parameters
# If passed function is an actual Python function, it will be executed with Placeholders as inputs.
# All layers functions call this at the end.
# BUGBUG: does not actually exist yet, faking it
# BUGBUG: should create a new object, but does it in-place instead. Works for current usage, but should be fixed.
# BUGBUG: using combine causes an error ater, so the name actually does not get changed
# BUGBUG: combine like this won't work for functions with multiple outputs (LSTM)
def Block(f, op_name, members={}):
    '''
    Experimental code. Don't assume it does anything.
    '''

    #from inspect import isfunction
    #if isfunction(f):
    #    f = Function(f, members)

    #f = combine([f], op_name)  # 'combine' to create a separate identity so we can reassign the debug name --BUGBUG: "Unknown DataType"
    #_name_node(f, op_name) ; _extend_Function(f)  # debugging

    # BUGBUG: alias() does not work. Once I add this, some clone() call no longer finds the placeholder
    #if len(p) != len(p1) or len(a) != len(a1):
    #    raise AssertionError('')

    for key in members:   # f.__dict__.update(members)
        f.__dict__[key] = members[key]
    return f


def BlockFunction(op_name, name):
    '''
    Same as @Function, but wrap the content into an as_block().
    '''
    return lambda f: Function(f, make_block=True, op_name=op_name, name=name)
    # TODO: bring this ^^ back after as_block works, and then undo the x_last hack
    # BUGBUG: Assumed only be used by recurrent cells to fix their ordering, which fails.
    #return Function(f)  # BUGBUG: causes random axis inference errors


def _inject_name(f, name):
    '''
    Call this at the end of any layer or block that takes an optional name argument.
    '''
    if name:
        #f = combine(f.outputs, name=name)
        # BUGBUG: will not work for sparse data, and not for tuple-valued Functions
        f = alias(f, name=name)
    return f


# TODO: Move this into the lower layer where these are defined.
# some mappings--these currently exist only so that I can name the nodes for debugging
def Parameter(shape, init, dtype=default_override_or(np.float32), name=''):
    '''
    Constructs a Parameter variable.
    '''
    pure = get_default_override(None, pure=default_override_or(False))
    if pure:
        raise TypeError('parameters cannot be created inside a @Function def')
    dtype = get_default_override(Parameter, dtype=dtype)
    init = _initializer_for(init)
    p = parameter(shape, init=init, dtype=dtype, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'parameter')   # these are factory methods for things with state

def Constant(value, shape=None, dtype=default_override_or(np.float32), name=''):
    '''
    Constructs a Variable object that is constant.
    '''
    dtype = get_default_override(Constant, dtype=dtype)
    p = constant(value, shape=shape, dtype=dtype, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'constant')   # these are factory methods for things with state

# TODO: this function should not be necessary anymore
def Input(shape, dtype=default_override_or(np.float32), needs_gradient=True, is_sparse=False,
          dynamic_axes=Axis.default_input_variable_dynamic_axes(), name=''):
    '''
    Constructs an Input variable.
    '''
    dtype = get_default_override(Input, dtype=dtype)
    return _name_node(input_variable(shape=shape, dtype=dtype, needs_gradient=needs_gradient, is_sparse=is_sparse,
                                     dynamic_axes=dynamic_axes, name=name), 'input')

def Placeholder(shape=None, dynamic_axes=None, is_sparse=False, name='placeholder'):
    '''
    Constructs a Placeholder variable.
    '''
    #p = placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, is_sparse=is_sparse, name=name) # TODO: use (*args, **kwargs)?
    # BUGBUG: placeholder does not know is_sparse
    p = placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, name=name) # TODO: use (*args, **kwargs)?
    _name_node(p, name)
    if _trace_layers:
        print("new " + _node_description(p))
    return p

def ForwardDeclaration(name='forward_declaration'):
    '''
    Helper for recurrent network declarations.
    Returns a Placeholder variable with an added method resolve_to() to be called
    at the end to close the loop.
    '''
    var_fwd = Placeholder(name=name)
    def resolve_to(var):
        from cntk import cntk_py
        if isinstance(var, cntk_py.Function):
            var.replace_placeholders({var_fwd: var.output})  # resolves var_fwd := var
        else:
            var.owner.replace_placeholders({var_fwd: var})   # resolves var_fwd := var
    var_fwd.resolve_to = resolve_to
    return var_fwd

# TODO: This should become a C++-side Function, e.g. like sigmoid
@Function
def identity(keep):
    '''
    Identity function.
    There is no factory for it because there is only one identity function.
    '''
    return combine([keep])


def Stabilizer(steepness=4, enable_self_stabilization=default_override_or(True), name=''):
    '''
    Layer factory function to create a Droppo stabilizer.
    This takes enable_self_stabilization as a flag that allows to disable itself. Useful if this is a global default.
    '''

    enable_self_stabilization = get_default_override(Stabilizer, enable_self_stabilization=enable_self_stabilization)

    if not enable_self_stabilization: # disabled (typically through global option; otherwise one would not call this in the first place)
        return identity

    # parameters bound to this Function
    param = Parameter((), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    # BUGBUG: dimension ^^ should be (), a scalar
    # TODO: compute this strange value directly in Python
    # TODO: implement softplus non-linearity in C++ for stability
    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness
    beta = log (1 + exp (steepness * param)) * (1 / steepness)   # perf BUGBUG: "log() / steepness" should optimize to the samething   --TODO: change in Python

    # expression
    @BlockFunction('Stabilizer', name)
    def stabilize(x):
        return beta * x

    #stabilize = _inject_name(stabilize, name)

    return Block(stabilize, 'Stabilizer', Record(beta=beta))


def _RecurrentBlock(type, shape, cell_shape, activation, use_peepholes,
                    init, init_bias,
                    enable_self_stabilization,
                    name=''):
    '''
    Helper to create a recurrent block of type 'LSTM', 'GRU', or RNNUnit.
    '''

    has_projection = cell_shape is not None

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape
    if len(shape) != 1 or len(cell_shape) != 1:
        raise ValueError("LSTM: shape and cell_shape must be vectors (rank-1 tensors)")
        # otherwise we'd need to fix slicing and Param initializers

    stack_axis = -1  # for efficient computation, we stack multiple variables (along the fastest-changing one, to match BS)
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
    H  = Parameter(shape     + cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden
    H1 = Parameter(shape     + cell_shape,           init=init,      name='H1') if type == 'GRU' else None  # hidden-to-hidden
    Ci = Parameter(            cell_shape,           init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,           init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,           init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init) if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization)
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization)
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization)
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization)

    # define the model function itself
    # general interface for Recurrence():
    #   (all previous outputs delayed, input) --> (outputs and state)
    # where
    #  - the first output is the main output, e.g. 'h' for LSTM
    #  - the remaining outputs, if any, are additional state
    #  - if for some reason output != state, then output is still fed back and should just be ignored by the recurrent block

    # TODO: rename all x_last back to x once parameter ordering works
    # LSTM model function
    # in this case:
    #   (dh, dc, x_last) --> (h, c)
    #@BlockFunction(type, name)
    #def lstm(dh, dc, x_last):
    #    x = x_last
    def lstm(dh, dc, x):
    # BUGBUG: now fails with Python crashing, likely the ref-count issue

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dhs, H)

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
    #   (dh, x_last) --> (h)
    # e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    # TODO: Is this the same definition as NVidia's? Should we enable multiple definitions of this?
    # BUGBUG: gru(x_last,dh,dc) passes, too. Since 'dc' is not referenced, it is just ignored. Also when routing it through combine().
    #          This may have changed with as_block(), which cannot handle unused inputs. TODO: test this.
    #@BlockFunction(type, name)
    def gru(dh, x_last):
        x = x_last

        dhs = Sdh(dh)  # previous value, stabilized
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        projx3 = b + times(x, W)
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

    #@BlockFunction(type, name)
    def rnn(dh, x_last):
        x = x_last
        dhs = Sdh(dh)  # previous value, stabilized
        ht = activation (times(x, W) + times(dhs, H) + b)
        h = times(Sht(ht), Wmr) if has_projection else \
            ht
        return Function.NamedOutput(h=h)

    function = {
        'RNNUnit': rnn,
        'GRU':     gru,
        'LSTM':    lstm
    }[type]

    # return the corresponding lambda as a CNTK Function
    function = BlockFunction(type, name)(function)

    # return the corresponding lambda as a CNTK Function

    # BUGBUG: This cannot work for the tuple-valued LSTM Function
    #function = _inject_name(function, name)

    return Block(function, type)


def LSTM(shape, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name=''):
    '''
    Layer factory function to create an LSTM block for use inside a recurrence.
    Returns a function (prev_h, prev_c, input) -> h).
    '''

    activation                = get_default_override(RNNUnit, activation=activation)
    use_peepholes             = get_default_override(LSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(LSTM, init=init)
    init_bias                 = get_default_override(LSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(LSTM, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('LSTM', shape, cell_shape, activation=activation, use_peepholes=use_peepholes,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization, name=name)


# TODO: needs better name
def RNNUnit(shape, cell_shape=None, activation=default_override_or(sigmoid),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name=''): # (prev_h, x) -> (h)
    '''
    Layer factory function to create a plain RNN block for use inside a recurrence.
    Returns a function (prev_h, input) -> h):
     h = activation (W * input + R * prev_h + b)
    '''

    activation                = get_default_override(RNNUnit, activation=activation)
    init                      = get_default_override(RNNUnit, init=init)
    init_bias                 = get_default_override(RNNUnit, init_bias=init_bias)
    enable_self_stabilization = get_default_override(RNNUnit, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('RNNUnit', shape, cell_shape, activation=activation, use_peepholes=False,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization, name=name)


def GRU(shape, cell_shape=None, activation=default_override_or(tanh),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name=''): # (prev_h, x) -> (h)
    '''
    Layer factory function to create a GRU block for use inside a recurrence.
    Returns a function (prev_h, input) -> h).
    '''

    activation                = get_default_override(GRU, activation=activation)
    init                      = get_default_override(GRU, init=init)
    init_bias                 = get_default_override(GRU, init_bias=init_bias)
    enable_self_stabilization = get_default_override(GRU, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock ('GRU', shape, cell_shape, activation=activation, use_peepholes=False,
                            init=init, init_bias=init_bias,
                            enable_self_stabilization=enable_self_stabilization, name=name)

# TODO in C++ code after next update:
#  - elementwise: sequence broadcasting for all elementwise operations, as a V2-V1 compilation step
