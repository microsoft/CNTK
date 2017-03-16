# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
blocks -- basic building blocks that are semantically not layers (not used in a layered fashion)
          e.g. the LSTM
'''

from __future__ import division
import numpy as np
from cntk import parameter, constant, input_variable, placeholder_variable, combine, alias, sequence
from cntk.axis import Axis
from cntk.ops import times, slice, sigmoid, tanh, log, exp, softplus, past_value, future_value
from cntk.utils import Record, Signature
from cntk.internal import _as_tuple
from cntk.initializer import glorot_uniform
from _cntk_py import InferredDimension
from cntk.default_options import *

from cntk.ops.functions import Function

_INFERRED = (InferredDimension,)  # as a tuple, makes life easier

# call this for all untested branches
def UntestedBranchError(name):
    raise NotImplementedError("Untested code branch: " + name)

# create the complete initializer for a given 'init' parameter, to pass to parameter()
# This is called from Parameter() and every place that injects rank parameters.
# It does a few things:
#  - maps init_default_override_or_glorot_uniform to default  --TODO: we should have a global setting for that
#  - creates a new initializer object from an existing one, while updating members
# TODO: remove default resolution, only make this a conversion; then rename
def _initializer_for(init, rank_params=None):
    if init is None:
        raise ValueError("init parameter cannot be None")

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

def BlockFunction(op_name, name):
    '''
    Decorator for defining a @Function as a BlockFunction. Same as @Function, but wrap the content into an as_block().
    '''
    return lambda f: Function(f, make_block=True, op_name=op_name, name=name)

def _inject_name(f, name):
    '''
    Call this at the end of any layer or block that takes an optional name argument.
    '''
    if name:
        if len(f.outputs) == 1:
            f = alias(f, name=name)
        else:
            f = combine(list(f.outputs), name=name) # BUGBUG: Does this actually name things?
    return f

# TODO: Move this into the lower layer where these are defined.
# some mappings--these currently exist only so that I can name the nodes for debugging
def Parameter(shape, init, dtype=default_override_or(np.float32), name=''):
    '''
    Parameter(shape, init, dtype=np.float32, name='')

    Constructs a Parameter variable.

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        init (scalar or NumPy array or :mod:`cntk.initializer`): initial value of weights `W`
        dtype (np.dtype, defaults to np.float32): data type
        name (str, defaults to ''): the name of the Function instance in the network
    '''
    
    pure = get_default_override(None, pure=default_override_or(False))
    if pure:
        raise TypeError('parameters cannot be created inside a @Function def')
    dtype = get_default_override(Parameter, dtype=dtype)
    init = _initializer_for(init)
    return parameter(shape, init=init, dtype=dtype, name=name)

def Constant(value, shape=None, dtype=default_override_or(np.float32), name=''):
    '''
    Constant(value, shape=None, dtype=np.float32, name='')

    Constructs a Variable object that is constant.

    Args:
        value (object): the object you want to make constant
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        dtype (np.dtype, defaults to np.float32): data type
        name (str, defaults to ''): the name of the Function instance in the network
    '''

    dtype = get_default_override(Constant, dtype=dtype)
    return constant(value, shape=shape, dtype=dtype, name=name)

# TODO: this function should not be necessary anymore
def Input(shape, dtype=default_override_or(np.float32), needs_gradient=True, is_sparse=False,
          dynamic_axes=Axis.default_input_variable_dynamic_axes(), name=''):
    '''
    Input(shape, dtype=np.float32, needs_gradient=True, is_sparse=False, dynamic_axes=Axis.default_input_variable_dynamic_axes(), name='')

    Constructs an Input variable.

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        dtype (np.dtype, defaults to np.float32): data type
        needs_gradient (bool, defaults to `True`):
        is_sparse (bool, defaults to `False`):
        dynamic_axes (object, Axis.default_input_variable_dynamic_axes):
        name (str, defaults to ''): the name of the Function instance in the network
        
    '''

    dtype = get_default_override(Input, dtype=dtype)
    return input_variable(shape=shape, dtype=dtype, needs_gradient=needs_gradient, is_sparse=is_sparse,
                          dynamic_axes=dynamic_axes, name=name)

def Placeholder(shape=None, dynamic_axes=None, is_sparse=False, name='placeholder'):
    '''
    Placeholder(shape=None, dynamic_axes=None, is_sparse=False, name='placeholder')

    Constructs a Placeholder variable.

    Args:
        shape (`int` or `tuple` of `ints`, defaults to `None`): vector or tensor dimension of the output of this layer
        dynamic_axes (object, defaults to `None`):
        is_sparse (bool, defaults to `False`):
        name (str, defaults to 'placeholder'): the name of the Function instance in the network
        
    '''

    if shape is not None or dynamic_axes is not None or is_sparse is not None:
        import warnings
        warnings.warn('Placeholder() no longer requires shapes, axes, or spares to be specified. Please just remove the arguments.', DeprecationWarning)
    return placeholder_variable(name=name)
    # TODO: delete these vv once confirmed that this is indeed not used anymore
    #p = placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, is_sparse=is_sparse, name=name) # TODO: use (*args, **kwargs)?
    # BUGBUG: placeholder does not know is_sparse
    #return placeholder_variable(shape=shape, dynamic_axes=dynamic_axes, name=name) # TODO: use (*args, **kwargs)?

def ForwardDeclaration(name='forward_declaration'):
    '''
    Helper for recurrent network declarations.
    Returns a Placeholder variable with an added method resolve_to() to be called
    at the end to close the loop.
    '''
    var_fwd = Placeholder(name=name)
    def resolve_to(var):
        from cntk import cntk_py
        #if isinstance(var, cntk_py.Function):
        #    var.replace_placeholders({var_fwd: var.output})  # resolves var_fwd := var
        #else:
        # TODO: ^^ should no longer be needed; delete once confirmed
        var.owner.replace_placeholders({var_fwd: var})   # resolves var_fwd := var
    var_fwd.resolve_to = resolve_to
    return var_fwd

@Function
def identity(keep):
    '''
    Identity function.
    There is no factory for it because there is only one identity function.
    '''
    # Note: We cannot use alias() here since parameter-shape inference cannot be done through alias().
    return combine([keep])


def Stabilizer(steepness=4, enable_self_stabilization=default_override_or(True), name=''):
    '''
    Stabilizer(steepness=4, enable_self_stabilization=True, name='')

    Layer factory function to create a `Droppo self-stabilizer <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf>`_.
    It multiplies its input with a scalar that is learned.

    This takes `enable_self_stabilization` as a flag that allows to disable itself. Useful if this is a global default.

    Note: Unlike the original paper, which proposed a linear or exponential scalar,
    CNTK uses a sharpened Softplus: 1/steepness ln(1+e^{steepness*beta}).
    The softplus behaves linear for weights around and above 1 (like the linear scalar) while guaranteeing
    positiveness (like the exponentional variant) but is also more robust by avoiding exploding gradients.

    Args:
        steepness (`int`, defaults to 4):
        enable_self_stabilization (bool, defaults to `False`): a flag that allows to disable itself. Useful if this is a global default
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function
    '''

    enable_self_stabilization = get_default_override(Stabilizer, enable_self_stabilization=enable_self_stabilization)

    if not enable_self_stabilization: # disabled (typically through global option; otherwise one would not call this in the first place)
        return identity

    # parameters bound to this Function
    init_param = np.log(np.exp(steepness) -1) / steepness  # initialize so that factor is initially 1 (has no effect)
    param = Parameter((), init=init_param, name='alpha')
    beta = softplus(param, steepness=steepness)

    # expression
    @BlockFunction('Stabilizer', name)
    def stabilize(x):
        return beta * x

    return stabilize


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
        raise ValueError("%s: shape and cell_shape must be vectors (rank-1 tensors)" % type)
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

    Wmr = Parameter(cell_shape + shape, init=init, name='P') if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dh_stabilizer')
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dc_stabilizer')
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='c_stabilizer')
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='P_stabilizer')

    # define the model function itself
    # general interface for Recurrence():
    #   (all previous outputs delayed, input) --> (outputs and state)
    # where
    #  - the first output is the main output, e.g. 'h' for LSTM
    #  - the remaining outputs, if any, are additional state
    #  - if for some reason output != state, then output is still fed back and should just be ignored by the recurrent block

    # LSTM model function
    # in this case:
    #   (dh, dc, x) --> (h, c)
    def lstm(dh, dc, x):

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

    # GRU model function
    # in this case:
    #   (dh, x) --> (h)
    # e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    def gru(dh, x):

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
        # h(t) = (1 - i(t) .* h'(t)) + i(t) .* h(t-1)                     --TODO: need to confirm bracketing with NVIDIA

        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        return Function.NamedOutput(h=h)

    def rnn(dh, x):
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
    return BlockFunction(type, name)(function)


def LSTM(shape, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name=''):
    '''
    LSTM(shape, cell_shape=None, activation=tanh, use_peepholes=False, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create an LSTM block for use inside a recurrence.

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): 
        activation (:class:`~cntk.ops.functions.Function`, defaults to tanh): function to apply at the end, e.g. `relu`
        use_peepholes (bool, defaults to `False`):
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`):
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function (prev_h, prev_c, input) -> h)
    '''

    activation                = get_default_override(RNNUnit, activation=activation)
    use_peepholes             = get_default_override(LSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(LSTM, init=init)
    init_bias                 = get_default_override(LSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(LSTM, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('LSTM', shape, cell_shape, activation=activation, use_peepholes=use_peepholes,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)


# TODO: needs better name
def RNNUnit(shape, cell_shape=None, activation=default_override_or(sigmoid),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False),
            name=''): # (prev_h, x) -> (h)
    '''
    RNNUnit(shape, cell_shape=None, activation=sigmoid, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create a plain RNN block for use inside a recurrence.

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): 
        activation (:class:`~cntk.ops.functions.Function`, defaults to signmoid): function to apply at the end, e.g. `relu`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`):
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function (prev_h, input) -> h) where h = activation (W * input + R * prev_h + b)
    '''

    activation                = get_default_override(RNNUnit, activation=activation)
    init                      = get_default_override(RNNUnit, init=init)
    init_bias                 = get_default_override(RNNUnit, init_bias=init_bias)
    enable_self_stabilization = get_default_override(RNNUnit, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('RNNUnit', shape, cell_shape, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)


def GRU(shape, cell_shape=None, activation=default_override_or(tanh),
        init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
        enable_self_stabilization=default_override_or(False),
        name=''): # (prev_h, x) -> (h)
    '''
    GRU(shape, cell_shape=None, activation=tanh, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')

    Layer factory function to create a GRU block for use inside a recurrence.

    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): 
        activation (:class:`~cntk.ops.functions.Function`, defaults to tanh): function to apply at the end, e.g. `relu`
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`):
        name (str, defaults to ''): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function (prev_h, input) -> h)
    '''

    activation                = get_default_override(GRU, activation=activation)
    init                      = get_default_override(GRU, init=init)
    init_bias                 = get_default_override(GRU, init_bias=init_bias)
    enable_self_stabilization = get_default_override(GRU, enable_self_stabilization=enable_self_stabilization)

    return _RecurrentBlock('GRU', shape, cell_shape, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)
