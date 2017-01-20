# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# higher_order_functions -- higher-order functions, like Sequential() and Recurrence()

#import numpy as np
#import sys
#import os
#import time

from .utils import Record
from .ops import combine, delay, sequence
from .blocks import *
from .blocks import _initializer_for, _get_initial_state_or_default, _INFERRED, _inject_name

def Sequential(layers, name=''):
    '''
    Composite that applies a sequence of layers (or any functions) onto an input.
    Sequential ([F, G, H]) === F >> G >> H
    '''
    if not isinstance(layers, list): # to support nested lists, run every item recursively through Sequential()
        # TODO: Is this confusing w.r.t. tuple which is parallel and list which is sequential?
        return layers
    from functools import reduce
    layers = [Sequential(layer) for layer in layers] # expand all layers recursively
    composed_function = reduce(lambda f, g: f >> g, layers, identity)

    composed_function = _inject_name(composed_function, name)

    # TODO: wrap this in a BlockFunction as to enforce inputs = inputs of first function
    return Block(composed_function, 'Sequential', Record(layers=layers))

def For(range, constructor, name=''):
    '''
    Composite that applies a sequence of layers constructed with a constructor lambda(layer).
    E.g.
    For(range(3), lambda i: Dense(2000))
    For(range(3), lambda: Dense(2000))
    '''
    #from inspect import signature
    #takes_arg = len(signature(constructor).parameters) > 0
    # Python 2.7 support requires us to use getargspec() instead
    from inspect import getargspec
    takes_arg = len(getargspec(constructor).args) > 0
    # helper to call the layer constructor
    def call(i):
        if takes_arg:
            return constructor(i)  # takes an arg: pass it
        else:
            return constructor()   # takes no arg: call without, that's fine too
    layers = [call(i) for i in range]
    sequential = Sequential(layers)

    sequential = _inject_name(sequential, name)

    return Block(sequential, 'For', Record(layers=layers))

# legacy name--remove
def LayerStack(N, constructor):
    return For(range(N), constructor)

# TODO: allow to say sequential=False, axis=2, length=100, ... something like this
def RecurrenceFrom(over_function, go_backwards=default_override_or(False), return_full_state=False, name=''):
    '''
    Runs a function recurrently over a time sequence, with initial state.
    This form is meant for use in sequence-to-sequence scenarios.
    The difference to Recurrence() is that this returns a function that accepts the initial state as data argument(s).
    Initial state consists of N arguments, matching 'over'.
    '''

    go_backwards  = get_default_override(RecurrenceFrom, go_backwards=go_backwards)

    import types
    if isinstance(over_function, types.FunctionType):
        UntestedBranchError("RecurrenceFrom() over a Python function")
        over_function = Function(over_function)

    # get signature of cell
    _, *prev_state_args = over_function.signature

    if len(over_function.outputs) != len(prev_state_args):
        raise TypeError('RecurrenceFrom: number of state variables inconsistent between create_placeholder() and recurrent block')

    # function that this layer represents
    def _recurrence_from_n(x, *initial_state):

        # TODO: move this entire placeholder business to Function.__call__
        out_vars_fwd = [ForwardDeclaration(name=state_var.name) for state_var in prev_state_args] # create list of placeholders for the state variables

        # previous function; that is, past or future_value with initial_state baked in
        #prev_out_vars = [Delay(T = -1 if go_backwards else +1, initial_state=init)(out_var) for out_var, init in zip(out_vars_fwd, initial_state)]  # delay (state vars)
        # BUGBUG: This fails ^^ due to current as_block() bugs; can only use Python function for now:
        prev_out_vars = [delay(out_var, initial_state=init, time_step=-1 if go_backwards else +1) for out_var, init in zip(out_vars_fwd, initial_state)]  # delay (state vars)

        # apply the recurrent block ('over_function')
        out = over_function(x, *prev_out_vars)  # this returns a Function (x, previous outputs...) -> (state vars...)

        # connect the recurrent dependency
        for (var_fwd, var) in zip(out_vars_fwd, list(out.outputs)):
            #var.owner.replace_placeholders({var_fwd: var})  # resolves out_vars_fwd := state_vars
            var_fwd.resolve_to(var)
        #replacements = { var_fwd: var for (var_fwd, var) in zip(out_vars_fwd, list(out.outputs)) }
        #out.replace_placeholders(replacements)  # resolves out_vars_fwd := state_vars

        # var_fwd.resolve_as(var)  -->  var.owner.replace_placeholders({var_fwd: var})

        if not return_full_state:
            out = combine([out.outputs[0]])  # BUGBUG: Without combine(), it fails with "RuntimeError: Runtime exception". TODO: fix this inside Function(lambda)?

        return out

    # functions that this layer represents
    # The @Function pattern only supports fixed signatures, so we need one for each #states we support.
    def recurrence_from_1(x, h):
        return _recurrence_from_n(x, h)
    def recurrence_from_2(x, h, c):
        return _recurrence_from_n(x, h, c)
    def recurrence_from_3(x, h, c, a):
        return _recurrence_from_n(x, h, c, a)

    recurrence_from_functions = [recurrence_from_1, recurrence_from_2, recurrence_from_3]
    num_state_args = len(prev_state_args)
    if num_state_args == 0 or num_state_args > len(recurrence_from_functions):
        raise ValueError('RecurrenceFrom() currently supports recurrence with up to {} state variables'.format(len(recurrence_from_functions)))

    # this creates the CNTK Function
    recurrence_from = Function(recurrence_from_functions[num_state_args-1])

    recurrence_from = _inject_name(recurrence_from, name)

    return Block(recurrence_from, 'RecurrenceFrom', Record(over_function=over_function))

def Recurrence(over_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Runs a function recurrently over a time sequence.
    This form is meant for use in regular recurrent-model scenarios.
    ``initial_state`` must be a constant (or at least have known shape). To pass initial_state as a data input, use RecurrenceFrom() instead.
    TODO: Can bidirectionality be an option of this? bidirectional=True? What was the reason it cannot?
    '''

    go_backwards  = get_default_override(Recurrence, go_backwards=go_backwards)
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    import types
    if isinstance(over_function, types.FunctionType):
        over_function = Function(over_function)

    # get signature of cell
    _, *prev_state_args = over_function.signature

    if len(over_function.outputs) != len(prev_state_args):
        raise TypeError('Recurrence: number of state variables inconsistent between create_placeholder() and recurrent block')

    # initial state can be a single value or one per state variable (if more than one, like for LSTM)
    if isinstance(initial_state, tuple) and len(initial_state) == 1:
        initial_state = initial_state[0]
    if not isinstance(initial_state, tuple):
        # TODO: if initial_state is a CNTK Function rather than an initializer, then require to pass it multiple times; otherwise broadcast to all
        initial_state = tuple(initial_state for out_var in prev_state_args)

    # express it w.r.t. RecurrenceFrom
    recurrence_from = RecurrenceFrom(over_function, go_backwards, return_full_state) # :: (x, state seq) -> (new state seq)

    # function that this layer represents
    @Function
    def recurrence(x):
        return recurrence_from(x, *initial_state)

    recurrence = _inject_name(recurrence, name)

    return Block(recurrence, 'Recurrence', Record(over_function=over_function))

def Fold(over_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Like ``Recurrence()`` but returns only the final state.
    '''

    go_backwards  = get_default_override(Fold, go_backwards=go_backwards)
    initial_state = get_default_override(Fold, initial_state=initial_state)

    # get the scan function
    recurrence = Recurrence(over_function, go_backwards=go_backwards, initial_state=initial_state, return_full_state=return_full_state)

    # now take the last or first
    select = sequence.first if go_backwards else sequence.last
    fold = recurrence >> tuple(select for output in recurrence.outputs)

    fold = _inject_name(fold, name)

    return Block(fold, 'Fold', Record(over_function=over_function))

def UnfoldFrom(over_function, map_state_function=identity, until_predicate=None, length_increase=1, initial_state=None, name=''):
    '''
    Implements an unfold() operation. It creates a function that, starting with a seed input,
    applies 'over_function' repeatedly and emits the sequence of results. Depending on the recurrent block,
    it may have this form:
       `result = f(... f(f([g(input), initial_state])) ... )`
    or this form:
       `result = f(g(input), ... f(g(input), f(g(input), initial_state)) ... )`
    where `f` is `over_function`.
    An example use of this is sequence-to-sequence decoding, where `g(input)` is the sequence encoder,
    `initial_state` is the sentence-start symbol, and `f` is the decoder. The first
    of the two forms above is a plain sequence-to-sequence model where encoder output
    is the start state for the output recursion.
    The second form is an attention-based decoder, where the encoded input affects every application
    of `f` differently.
    '''

    import types
    if isinstance(map_state_function, types.FunctionType):
        map_state_function = Function(map_state_function)
    if isinstance(until_predicate, types.FunctionType):
        until_predicate = Function(until_predicate)

    @Function
    def unfold_from(input, dynamic_axes_like):
        # create a new axis if needed
        out_axis = dynamic_axes_like
        if length_increase != 1:
            factors = sequence.constant_with_dynamic_axes_like(length_increase, out_axis) # repeat each frame 'length_increase' times, on average
            out_axis = sequence.where(factors)  # note: values are irrelevant; only the newly created axis matters

        # BUGBUG: This will fail with sparse input.
        # nearly the same as RecurrenceFrom(); need to swap parameter order for either LSTM or decoder
        history_fwd = Placeholder(name='hook')  # TODO: change to ForwardDeclaration()
        prev_history = delay(history_fwd, initial_state=initial_state)
        z = over_function(prev_history, input)#,      out_axis)
        # apply map_state_function
        fb = map_state_function(z)
        # apply dynamic_axes_like
        from .utils import sanitize_input, typemap
        from _cntk_py import reconcile_dynamic_axis
        fb = typemap(reconcile_dynamic_axis)(sanitize_input(fb), sanitize_input(out_axis))
        z.replace_placeholders({history_fwd : fb.output})

        # apply until_predicate if given
        if until_predicate is not None:
            from cntk.ops.sequence import gather
            valid_frames = Recurrence(lambda x, h: (1-past_value(x)) * h, initial_state=1)(until_predicate(z))
            z = gather(z, valid_frames)

        return z

    unfold_from = _inject_name(unfold_from, name)

    return Block(unfold_from, 'UnfoldFrom', Record(over_function=over_function))
