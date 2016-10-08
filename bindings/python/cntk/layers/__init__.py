# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# cntk\layers

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

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
# TODO: move these out from examples
from examples.common.nn import slice, sigmoid, log, tanh, past_value, future_value, print_training_progress, negate

#### temporary layers lib, to be moved out
from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# "upgrade" a current Function to add additional operators and methods, as a temporary stopgap
# at end of each layer constructor, return _extend_Function(z, 'Type (for debugging)')
# use this until __call__ is implemented in Function()
# Also add the >> operator (forward function composition).
# Returns its arg to allow chaining.
def _extend_Function(f):
    class FunctionEx(f.__class__): 
        def __call__(self, *args):
            return _apply(self, _as_tuple(args))
        def __rshift__(self, other):
            return other(self)
        def _name(self):  # retrieve the debug name
            return _node_name(self)
    if hasattr(f, '__call__'):  # already extended: don't do it again
        return f
    f.__class__ = FunctionEx
    print("def {}".format(_node_description(f)))
    return f

# name and extend; in this order, so that _extend_Function can print a meaningful log message
def _name_and_extend_Function(f, name):
    _name_node(f, name)
    _extend_Function(f)

# give a new name to a function, by wrapping it
def _wrap_rename_Function(f, name):
     f = combine([f]) ; _name_and_extend_Function(f, name)  # 'combine' to create a separate identity so we can reassign the debug name
     return f

# monkey-patching some missing stuff
def __matmul__(a,b):  # TODO: define @ once we have Python 3.5
    return times(a,b)
#Function.__matmul__ = __matmul__  # should work in Python 3.5  --Function is not defined?

def _Infer(shape, axis):
    return Record(shape=_as_tuple(shape), axis=axis, with_shape = lambda new_shape: _Infer(new_shape, axis))

def _apply(f, args):
    import operator   # add()
    import functools  # reduce()
    from cntk.cntk_py import ParameterCloningMethod_Share
    # flatten args to a list. Note it may be a a tuple or even a nested tree of tuples, e.g. LSTM (x, (h, c))
    def flatten_tuple(args):
        if not isinstance(args, tuple): # not a tuple: singleton; create a singleton tuple
            return (args,)
        return functools.reduce(operator.add, [(flatten_tuple(item)) for item in args])
    args = list(flatten_tuple(args))
    def _output_of(arg):  # helper to get the output of an arg; use arg itself if no output() method (that'd be a Variable)
        try:
            return arg.output()
        except AttributeError:
            return arg  # Variables have no output()
    args = [_output_of(arg) for arg in args]
    placeholders = f.placeholders()  # f parameters to fill in
    #print (len(args))
    #print (len(placeholders))
    if len(args) != len(placeholders):
        raise TypeError("_apply ({}): number of arguments {} must match number of placeholders {}".format(_node_description(f), len(args), len(placeholders)))
    _function_name = _node_name(f)  # these are for logging/debugging only
    _function_description = _node_description(f)
    _arg_description = ", ".join([_node_name(f) for f in list(args)])
    f = f.clone(ParameterCloningMethod_Share)
    f.replace_placeholders(dict(zip(f.placeholders(), args)))
    #f = f.clone(dict(zip(placeholders, args)))
    # BUGBUG: need to get this to work, in conjunction with _Share
    _name_and_extend_Function(f, _function_name)
    print("{} = {} ({})".format(_node_description(f), _function_description, _arg_description))
    return f

# some mappings to BS format
def Parameter(shape, learning_rate_multiplier=1.0, init=None, init_value_scale=1, init_value=None, init_filter_rank=0, init_output_rank=1, init_from_file_path=None, init_on_cpu_only=True, random_seed=-1):
    return _name_node(parameter(shape), 'parameter')   # these are factory methods for things with state
def Input(*args, **kwargs):
    return _name_node(input_variable(*args, **kwargs), 'input')

def Placeholder(_inf, name='placeholder'):
    # BUGBUG: does not take a name parameter (but not really needed here)
    # BUGBUG: combine() does not work either, as it generates a Function, not a Variable
    p = placeholder_variable(shape=_as_tuple(_inf.shape), dynamic_axes=_inf.axis)
    _name_node(p, name)
    print("new " + _node_description(p))
    return p

# Sequential -- composite that applies a sequence of functions onto an input
# Sequential ([F, G, H]) === F >> G >> H
def Sequential(arrayOfFunctions, _inf):
    import functools  # reduce()
    apply_x = functools.reduce(lambda f, g: f >> g, arrayOfFunctions, Identity(_inf=_inf))
    apply_x = _wrap_rename_Function(apply_x, 'Sequential')
    return apply_x;

# need to define everything indented by 4

# Linear -- create a fully-connected linear projection layer
# Note: shape may describe a tensor as well.
# TODO: change to new random-init descriptor
def Linear(shape, _inf, bias=True, init='glorot_uniform', init_value_scale=1, input_rank=None, map_rank=None):
    out_shape = _as_tuple(shape)
    W = Parameter(_inf.shape + out_shape, init=init, init_value_scale=init_value_scale)
    b = Parameter(             out_shape, init='zero') if bias else None
    x = Placeholder(_inf=_inf, name='linear_arg')
    apply_x = __matmul__(x, W) + b if bias else \
              __matmul__(x, W)
    _name_and_extend_Function(apply_x, 'Linear')
    return apply_x
    # TODO: how to break after the else?

# Embedding -- create a linear embedding layer
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
    _ = Placeholder(_inf=_inf, name='embedding_arg')
    apply_x = __matmul__(E, _) if transposed else \
            __matmul__(_, E)     # x is expected to be sparse one-hot
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

def Identity(_inf):
    x = Placeholder(_inf=_inf, name='identity_arg')
    #apply_x = combine([x])  # BUGBUG: not working
    apply_x = x + 0  # this fakes combine()
    _name_and_extend_Function(apply_x, 'Identity')
    return apply_x

# TODO: For now, shape and cell_shape can only be rank-1 vectors
def LSTMBlock(shape, _inf, cell_shape=None, use_peepholes=False, init='glorot_uniform', init_value_scale=1, enable_self_stabilization=False): # (x, (h, c))
    has_projection = cell_shape is not None
    has_aux = False

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape

    #stack_axis = -1  # 
    stack_axis = 0  # BUGBUG: should be -1, i.e. the fastest-changing one, to match BS
    # determine stacking dimensions
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[0]
    cell_shape_list[stack_axis] = stacked_dim*4
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    B  = Parameter(             cell_shape_stacked, init_value=0)       # a bias
    W  = Parameter(_inf.shape + cell_shape_stacked, init=init, init_value_scale=init_value_scale)                             # input
    A  = Parameter(_inf.shape + cell_shape_stacked, init=init, init_value_scale=init_value_scale) if has_aux else None        # aux input (optional)
    H  = Parameter(shape      + cell_shape_stacked, init=init, init_value_scale=init_value_scale)                             # hidden-to-hidden
    Ci = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = ParameterTensor (cell_shape + shape, init=init, init_value_scale=init_value_scale) if has_projection else None  # final projection

    Sdh = Stabilizer(_inf=_inf.with_shape(     shape)) if enable_self_stabilization else Identity(_inf=_inf.with_shape(     shape))
    Sdc = Stabilizer(_inf=_inf.with_shape(cell_shape)) if enable_self_stabilization else Identity(_inf=_inf.with_shape(cell_shape))
    Sct = Stabilizer(_inf=_inf.with_shape(cell_shape)) if enable_self_stabilization else Identity(_inf=_inf.with_shape(cell_shape))
    Sht = Stabilizer(_inf=_inf.with_shape(     shape)) if enable_self_stabilization else Identity(_inf=_inf.with_shape(     shape))

    def create_hc_placeholder():
        return (Placeholder(_inf=_inf.with_shape(shape), name='hPh'), Placeholder(_inf=_inf.with_shape(cell_shape), name='cPh')) # (h, c)

    # parameters to model function
    x = Placeholder(_inf=_inf, name='lstm_block_arg')
    prev_state = create_hc_placeholder()

    # formula of model function
    dh, dc = prev_state

    dhs = Sdh(dh)  # previous values, stabilized
    dcs = Sdc(dc)
    # note: input does not get a stabilizer here, user is meant to do that outside

    # projected contribution from input(s), hidden, and bias
    proj4 = B + times(x, W) + times(dhs, H) + times(aux, A) if has_aux else \
            B + times(x, W) + times(dhs, H)

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
    _print_node(h)  # this looks right
    _name_node(c, 'c')

    # return to caller a helper function to create placeholders for recurrence
    apply_x_h_c = combine ([h, c])
    apply_x_h_c.create_placeholder = create_hc_placeholder
    _name_and_extend_Function(apply_x_h_c, 'LSTMBlock')
    return apply_x_h_c

def Recurrence(over=None, _inf=None, go_backwards=False):
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
    repl_list = { value_forward: value.output() for (value_forward, value) in list(zip(list(prev_state_forward), list(prev_state))) }
    f_x_h_c.replace_placeholders(repl_list)  # binds _h_c := prev_state
    apply_x = combine([h.owner()])     # the Function that yielded 'h', so we get to know its inputs
    # apply_x is a Function x -> h
    _name_and_extend_Function(apply_x, 'Recurrence')
    _print_node(apply_x)
    return apply_x
