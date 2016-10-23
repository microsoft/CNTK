# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# blocks -- basic building blocks that are semantically not layers (not used in a layered fashion)
#           e.g. the LSTM

# TODO: further clean up the dependencies
import numpy as np
#import sys
#import os
#import time
from cntk import parameter, constant, input_variable, placeholder_variable, combine, alias
from cntk.ops import times, slice, sigmoid, tanh, log, exp, past_value, future_value
from cntk.utils.debughelpers import _name_node, _node_name, _node_description, _log_node
from cntk.utils import Record, _as_tuple
from cntk.initializer import glorot_uniform
from _cntk_py import InferredDimension

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

# This record contains the defaults for a number of optional parameters to layers.
# These can be overwritten temporarily by saying
#    with default_options(init=..., ...):
#        # code block within which the changed defaults are active
_current_default_options = Record(
    init=glorot_uniform(),
    activation=None,                  # Dense() and Convolution() have no activation by default
    pad=False, # BUGBUG: not done for pooling at present. Need a special default? How to name?
    # ^^ This should be addressed by allowing configs per layer type.
    #    To be addressed as a per-layer default. See default_options below.
    bias=True,
    init_bias=0,
    enable_self_stabilization=False,  # Stabilizer() and LSTM()
    initial_state=None                # Recurrence()
)

_default_sentinel           = Record() # This is a singleton sentinel value we recognize and replace in _initializer_for()
_default_sentinel_init      = Record() # use different ones for init andinit_bias so we can distinguish them in _initializer_for()
_default_sentinel_init_bias = Record()
# in function signatures we use symbols that indicate the default default in their name
init_default_or_glorot_uniform             = _default_sentinel_init
activation_default_or_None                 = _default_sentinel
init_bias_default_or_0                     = _default_sentinel_init_bias
bias_default_or_True                       = _default_sentinel
pad_default_or_False                       = _default_sentinel
enable_self_stabilization_default_or_False = _default_sentinel
initial_state_default_or_None              = _default_sentinel

# check whether a parameter is a default
# This is meant to be used by implementations of layers that take default values that may default to default-defaults.
def _is_given(p):
    return p is not _default_sentinel and p is not _default_sentinel_init and p is not _default_sentinel_init_bias

# scope guard for overriding defaults
# with default_options(activation=relu, init=he_normal(), pad=True):
#     model = Convolution((3,3), 32) >> Convolution((3,3), 64)  # will have relu activation and padding
# to limit it to a specific operation or set of operations, use e.g.
# with default_options(activation=relu, init=he_normal()):
#     with default_options_for(Convolution, pad=True):
#         ...
# TODO: actually implement this
class _OptionsStack: # implement Python's 'with' protocol
    # constructor prepares the updated record of defaults and remembers it
    def __init__(self, layer_subset=None, **kwargs):
        if layer_subset:
            raise NotImplementedError('default_options not yet implemented per layer')
        new_options = kwargs
        # TODO: layer subset
        merged_options = _current_default_options.__dict__
        # we merge the newly provided options with the current defaults
        # Only names that already exist may be used (cannot create new default variables).
        # TODO: we may consider a more generic mechanism where one can overwrite all, if Python allows that.
        for key in new_options:
            try:
                merged_options[key]
            except:
                raise TypeError("default_options() got an unexpected keyword argument '{}'".format(key))
            merged_options[key] = new_options[key]
        self.new_default_options = Record(**merged_options) # this is the new options record that entering the with section will activate
    # entering with block: replaces the current defaults with the new ones, after remembering the old ones
    def __enter__(self):
        global _current_default_options
        self.saved_default_options = _current_default_options
        _current_default_options = self.new_default_options
    # exiting with block: restore previous remembered defaults
    def __exit__(self, type, value, traceback):
        global _current_default_options
        _current_default_options = self.saved_default_options

def default_options(**kwargs):
    return _OptionsStack(None, **kwargs)

def default_options_for(layer_subset, **kwargs):
    if not isinstance(layer_subset, list):
        layer_subset = [layer_subset]
    return _OptionsStack(layer_subset, **kwargs)

# resolve activation option against current default
def _resolve_activation(activation):
    # if none is passed to caller then use the default
    if activation is _default_sentinel:
        activation = _current_default_options.activation
    # activation=None is implemented as activation=identity
    if activation is None:
        activation = identity
    return activation

# create the complete initializer for a given 'init' parameter, to pass to parameter()
# This is called from Parameter() and every place that injects rank parameters.
# It does a few things:
#  - maps init_default_or_glorot_uniform to default  --TODO: we should have a global setting for that
#  - creates a new initializer object from an existing one, while updating members
def _initializer_for(init, rank_params=None):
    if init is None:
        raise ValueError("init parameter cannot be None")

    # if default then select
    if init is _default_sentinel_init:
        init = _current_default_options.init
    elif init is _default_sentinel_init_bias:
        init = _current_default_options.init_bias

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
    for key in members:   # self.__dict__.update(args_dict)
        f.__dict__[key] = members[key]
    return f

# some mappings--these currently exist only so that I can name the nodes for debugging
def Parameter(shape, init, name=''):
    init = _initializer_for(init)
    p = parameter(shape, init=init, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'parameter')   # these are factory methods for things with state

def Constant(init, shape=None, name=''):
    p = constant (init, shape, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'constant')   # these are factory methods for things with state

def Input(*args, **kwargs):
    return _name_node(input_variable(*args, **kwargs), 'input')

def Placeholder(name='placeholder'):
    p = placeholder_variable(name=name) # TODO: use (*args, **kwargs) once got rid of _inf
    _name_node(p, name)
    if _trace_layers:
        print("new " + _node_description(p))
    return p

# TODO: this is a left-over until inference works
def PlaceholderWithShape(_inf, name='placeholder'):
    p = placeholder_variable(shape=_as_tuple(_inf.shape), dynamic_axes=_inf.axis, name=name)
    _name_node(p, name)
    if _trace_layers:
        print("new " + _node_description(p))
    return p

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

# TODO: add a flag enable_self_stabilizer (maybe rename it) default to True, overridable by default_options
# This takes enable_self_stabilization as a flag that allows to disable itself. Useful if this is a global default.
def Stabilizer(steepness=4, enable_self_stabilization=enable_self_stabilization_default_or_False):
    enable_self_stabilization = enable_self_stabilization if _is_given(enable_self_stabilization) else _current_default_options.enable_self_stabilization
    if not enable_self_stabilization: # disabled (typically through global option)
        return identity
    #UntestedBranchError("Stabilizer")
    # currently fails with "RuntimeError: The 1 leading dimensions of the right operand with shape [1] do not match the left operand's trailing dimensions with shape [943]"

    # When applied to a known input, it works with linear stabilizer, but not with softplus,
    # which is explaninable as time stamp not being updated in model update.

    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness

    # parameters bound to this Function
    param = Parameter((1), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    #param = Parameter((1), init=1, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
    # TODO: compute this strange value directly in Python

    # expression
    x = Placeholder(name='stabilizer_arg')
    # TODO: risk of confusion; can these functions be namespaced?
    #beta = log (1 + exp (steepness * param)) * (1 / steepness)   # perf BUGBUG: "log() / steepness" should optimize to the samething
    #from cntk import log, exp
    #beta = log (param - 0.99537863 + 2.7182818284590452353602874713527)   # HAS NO EFFECT
    #beta = log (exp (param))   # HAS NO EFFECT
    #beta = (exp (param))   # HAS NO EFFECT
    #beta = steepness * param * (1/steepness) # HAS NO EFFECT
    beta = param # TODO: replace by softplus once the time-stamp issue is gone. Softplus works better.
    apply_x = beta * x
    return Block(apply_x, 'Stabilizer', Record(beta=beta))

def LSTM(shape, cell_shape=None, use_peepholes=False,
         init=init_default_or_glorot_uniform, init_bias=init_bias_default_or_0,
         enable_self_stabilization=enable_self_stabilization_default_or_False): # (x, (h, c))

    enable_self_stabilization = enable_self_stabilization if _is_given(enable_self_stabilization) else _current_default_options.enable_self_stabilization
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
    if Wmr:
        UntestedBranchError("LSTM, projection")

    Sdh = Stabilizer() if enable_self_stabilization else identity
    Sdc = Stabilizer() if enable_self_stabilization else identity
    Sct = Stabilizer() if enable_self_stabilization else identity
    Sht = Stabilizer() if enable_self_stabilization else identity

    def create_hc_placeholder():
        return (placeholder_variable(shape, name='hPh'), placeholder_variable(cell_shape, name='cPh')) # (h, c)

    # parameters to model function
    x = placeholder_variable(name='lstm_block_arg')
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
