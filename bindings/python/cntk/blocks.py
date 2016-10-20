# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# blocks -- basic building blocks that are semantically not layers (not used in a layered fashion)
#           e.g. the LSTM

# TODO: further clean up the dependencies
import numpy as np
import sys
import os
import time
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

_default_initializer = glorot_uniform()
_INFERRED = (InferredDimension,)  # as a tuple, makes life easier

def UntestedBranchError(name):
    # pass
    raise NotImplementedError("Untested code branch: " + name)

# "upgrade" a current Function to add additional operators and methods, as a temporary stopgap
# at end of each layer constructor, return _extend_Function(z, 'Type (for debugging)')
# use this until __call__ is implemented in Function()
# Also add the >> operator (forward function composition).
# Returns its arg to allow chaining.
def _extend_Function(f):
    class FunctionEx(f.__class__): 
        # BUGBUG: Somehow we don't get here anymore. It's fine for now.
        def __dummy__():
            pass
        #def __call__(self, *args):
        #    f = self
        #    _function_name = _node_name(f)  # these are for logging/debugging only
        #    _function_description = _node_description(f)
        #    _arg_description = ", ".join([_node_name(f) for f in list(args)])
        #    #f = super(Function, self).f()
        #    f = super(Function, f)(*args)
        #    _name_and_extend_Function(f, _function_name)
        #    if _trace_layers:
        #        print("{} = {} ({})".format(_node_description(f), _function_description, _arg_description))
        #    return f
        ## needed here to call into FunctionEx.__call__ instead of Function.__call__
        #def __rshift__(self, other):
        #    return other(self)
        def _name(self):  # retrieve the debug name
            return _node_name(self)
    if hasattr(f, '__dummy__'):  # already extended: don't do it again
        return f
    f.__class__ = FunctionEx
    if _trace_layers:
        print("def {}".format(_node_description(f)))
    return f

# name and extend; in this order, so that _extend_Function can print a meaningful log message
def _name_and_extend_Function(f, name=None):
    if name is not None:
        _name_node(f, name)
    _extend_Function(f)

# give a new name to a function, by wrapping it
def _wrap_rename_Function(f, name):
     f = combine([f]) ; _name_and_extend_Function(f, name)  # 'combine' to create a separate identity so we can reassign the debug name
     return f

# TODO: no longer used in this form
def __unused_apply(f, args):
    import operator   # add()
    import functools  # reduce()
    from cntk.ops.functions import CloneMethod
    # flatten args to a list. Note it may be a a tuple or even a nested tree of tuples, e.g. LSTM (x, (h, c))
    def flatten_tuple(args):
        if not isinstance(args, tuple): # not a tuple: singleton; create a singleton tuple
            return (args,)
        return functools.reduce(operator.add, [(flatten_tuple(item)) for item in args])
    args = list(flatten_tuple(args))
    # TODO: This should go into Function.replace_placeholders()
    def _output_of(arg):  # helper to get the output of an arg; use arg itself if no output() method (that'd be a Variable)
        try:
            return arg.output
        except AttributeError:
            return arg  # Variables have no output()
            #return arg.output  # Variables have no output()  --should work but doesn't
    args = [_output_of(arg) for arg in args]  # BUGBUG: without: "TypeError: cannot convert value of dictionary to CNTK::Variable "
    placeholders = f.placeholders  # f parameters to fill in
    if len(args) != len(placeholders):
        raise TypeError("_apply ({}): number of arguments {} must match number of placeholders {}".format(_node_description(f), len(args), len(placeholders)))
    _function_name = _node_name(f)  # these are for logging/debugging only
    _function_description = _node_description(f)
    _arg_description = ", ".join([_node_name(f) for f in list(args)])
    f = f.clone(CloneMethod.share, dict(zip(f.placeholders, args)))
    _name_and_extend_Function(f, _function_name)
    if _trace_layers:
        print("{} = {} ({})".format(_node_description(f), _function_description, _arg_description))
    return f

# some mappings--these currently exist only so that I can name the nodes for debugging
# TODO: random init parameters: init_filter_rank=0, init_output_rank=1, init_on_cpu_only=True, random_seed=-1
# init must be given
def Parameter(shape, init, name=''):
    if init is None:
        raise "Parameter: init cannot be None"
    p = parameter(shape, init=init, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'parameter')   # these are factory methods for things with state

def Constant(init, shape=None, name=''):
    p = constant (init, shape, name=name) # TODO: use (*args, **kwargs)
    return _name_node(p, 'constant')   # these are factory methods for things with state

def Input(*args, **kwargs):
    return _name_node(input_variable(*args, **kwargs), 'input')

def Placeholder(_inf=None, name='placeholder'):
    if _inf is None:
        p = placeholder_variable(name=name) # TODO: use (*args, **kwargs) once got rid of _inf
    else:  # BUGBUG: This code branch is only needed for Recurrence(), and will go away completely.
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
    _name_and_extend_Function(apply_x, 'Identity')
    return apply_x

# there is only one identity function
# TODO: This should become a C++-side Function, e.g. like sigmoid
identity = _Identity()

def Stabilizer(steepness=4):
    #UntestedBranchError("Stabilizer")
    # This was tested ad-hoc, and it works with linear stabilizer, but not with softplus,
    # which shows that it is not a bug in this code but maybe some time-stamp problem.

    # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
    # this behaves linear for weights around 1, yet guarantees positiveness

    # parameters bound to this Function
    param = Parameter((1), init=0.99537863, name='stabilizer_param')  # 1/steepness*ln (e^steepness-1) for steepness==4
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
    beta = param # TODO: replace by function above
    apply_x = beta * x
    _name_and_extend_Function(apply_x, 'Stabilizer')
    return apply_x

# TODO: For now, shape and cell_shape can only be rank-1 vectors
def LSTM(shape, cell_shape=None, use_peepholes=False, init=_default_initializer, init_bias=0, enable_self_stabilization=False): # (x, (h, c))
    has_projection = cell_shape is not None
    has_aux = False

    if has_aux:
        UntestedBranchError("LSTM, has_aux option")
    if has_projection:
        UntestedBranchError("LSTM, projection")

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape

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
        return (Placeholder(name='hPh'), Placeholder(name='cPh')) # (h, c)

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
    apply_x_h_c.create_placeholder = create_hc_placeholder
    _name_and_extend_Function(apply_x_h_c, 'LSTM')
    return apply_x_h_c
