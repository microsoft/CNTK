# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# blocks -- basic building blocks that are semantically not layers (not used in a layered fashion)
#           e.g. the LSTM
# TODO: This has become too large. Need to break out some locally used stuff into another module.

# TODO: further clean up the dependencies
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

# turn a Function into a Block, with a new name and an optional dictionary of named parameters
# All layers functions call this at the end.
# BUGBUG: does not actually exist yet, faking it
# BUGBUG: should create a new object, but does it in-place instead. Works for current usage, but should be fixed.
# BUGBUG: using combine causes an error ater, so the name actually does not get changed
# BUGBUG: combine like this won't work for functions with multiple outputs (LSTM)
def Block(f, op_name, members={}):
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

# use this for set_signature()
variable_of_type = Input

def variable_type_of(*args, **kwargs):
    return _name_node(input_variable(*args, **kwargs), 'input')

def Placeholder(shape=None, name='placeholder'):
    if shape is not None:
        p = placeholder_variable(shape=shape, name=name) # TODO: use (*args, **kwargs)?
    else:
        p = placeholder_variable(name=name) # TODO: use (*args, **kwargs)?
    _name_node(p, name)
    if _trace_layers:
        print("new " + _node_description(p))
    return p

# e.g.
# x, y = Placeholders(2)
def Placeholders(num_positional, *named_names):
    args = [Placeholder() for i in range(num_positional)]
    combined_args = combine(args) # create a compound that traverses in the defined order
    return tuple([output for output in combined_args.outputs])

#SymbolicArgument = Placeholder  # Placeholder is too overloaded; we should use this instead everywhere

# If we have C++-side Function identity, in same pattern as e.g. sigmoid, then that would suffice.
def _Identity(name='identity_arg'):
    x = Placeholder(name=name)
    apply_x = combine([x])
    # TODO: Let's not encourage users to use combine([f]) as a workaround for identity/pass, but rather have it as a first-class operator implemented that we then use. [Willi]
    #apply_x = alias(x) # TODO: does not work. Should it?
    #_name_and_extend_Function(apply_x, 'Identity')
    return Block(apply_x, 'Identity')

# there is only one identity function
# TODO: This should become a C++-side Function, e.g. like sigmoid
identity = _Identity()

# This takes enable_self_stabilization as a flag that allows to disable itself. Useful if this is a global default.
def Stabilizer(steepness=4, enable_self_stabilization=default_override_or(True)):

    enable_self_stabilization = get_default_override(Stabilizer, enable_self_stabilization=enable_self_stabilization)

    if not enable_self_stabilization: # disabled (typically through global option; otherwise one would not call this in the first place)
        return identity

    # parameters bound to this Function
    param = Parameter((1), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    #param = Parameter((1), init=1, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    # TODO: compute this strange value directly in Python

    # expression
    x = Placeholder(name='stabilizer_arg')

    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness
    # TODO: risk of confusion; can these functions be namespaced?
    beta = log (1 + exp (steepness * param)) * (1 / steepness)   # perf BUGBUG: "log() / steepness" should optimize to the samething
    apply_x = beta * x
    return Block(apply_x, 'Stabilizer', Record(beta=beta))

def LSTM(shape, cell_shape=None, use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False)): # (x, (h, c))

    use_peepholes             = get_default_override(LSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(LSTM, init=init)
    init_bias                 = get_default_override(LSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(LSTM, enable_self_stabilization=enable_self_stabilization)

    #use_peepholes             = use_peepholes             if _is_given(use_peepholes)             else _current_default_options.use_peepholes
    #enable_self_stabilization = enable_self_stabilization if _is_given(enable_self_stabilization) else _current_default_options.enable_self_stabilization

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
    stacked_dim = cell_shape_list[0]
    cell_shape_list[stack_axis] = stacked_dim*4
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked, init=init_bias, name='b')                              # a bias
    W  = Parameter(_INFERRED + cell_shape_stacked, init=init,      name='W')                              # input
    A  = Parameter(_INFERRED + cell_shape_stacked, init=init,      name='A') if has_aux else None         # aux input (optional)
    H  = Parameter(shape     + cell_shape_stacked, init=init,      name='H')                              # hidden-to-hidden
    Ci = Parameter(            cell_shape,         init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,         init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,         init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init) if has_projection else None  # final projection

    Sdh = Stabilizer() if enable_self_stabilization else identity
    Sdc = Stabilizer() if enable_self_stabilization else identity
    Sct = Stabilizer() if enable_self_stabilization else identity
    Sht = Stabilizer() if enable_self_stabilization else identity

    def create_hc_placeholder():
        # we pass the known dimensions here, which makes dimension inference easier
        return (Placeholder(shape=shape, name='hPh'), Placeholder(shape=cell_shape, name='cPh')) # (h, c)

    # parameters to model function
    x = Placeholder(name='lstm_block_arg')
    prev_state = create_hc_placeholder()

    # formula of model function
    dh, dc = prev_state

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

    # add peephole connection if requested
    def peep(x, c, C):
        return x + C * c if use_peepholes else x

    it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
    bit = it * tanh (bit_proj)                    # applied to tanh of input network

    ft = sigmoid (peep (ft_proj, dcs, Cf))        # forget-me-not gate(t)
    bft = ft * dc                                 # applied to cell(t-1)

    ct = bft + bit                                # c(t) is sum of both

    ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
    ht = ot * tanh (ct)                           # applied to tanh(cell(t))

    c = ct                                        # cell value
    h = times(Sht(ht), Wmr) if has_projection else \
        ht

    _name_node(h, 'h')
    if _trace_layers:
        _log_node(h)  # this looks right
    _name_node(c, 'c')

    # TODO: figure out how to do scoping, and also rename all the apply... to expression
    apply_x_h_c = combine ([h, c])
    # return to caller a helper function to create placeholders for recurrence
    # Note that this function will only exist in the object returned here, but not any cloned version of it.
    apply_x_h_c.create_placeholder = create_hc_placeholder
    #return Block(apply_x_h_c, 'LSTM') # BUGBUG: fails with "RuntimeError: A Function instance with more than one output cannot be implicitly converted to a Variable"
    return apply_x_h_c
