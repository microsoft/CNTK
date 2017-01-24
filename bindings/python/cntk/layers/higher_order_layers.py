# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# higher_order_functions -- higher-order functions, like Sequential() and Recurrence()

from ..utils import Record
from ..ops import combine, delay, sequence
from .blocks import *
from .blocks import _initializer_for, _get_initial_state_or_default, _INFERRED, _inject_name


def Sequential(layers, name=''):
    '''
    Layer factory function to create a composite that applies a sequence of layers (or any functions) onto an input.
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
    Layer factory function to create a composite that applies a sequence of layers constructed with a constructor lambda(layer).
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


def _sanitize_function(f):
    '''
    Helper to type-cast a Python function into a CNTK Function if not yet.
    '''
    import types
    if isinstance(f, types.FunctionType):
        f = Function(f)
    return f


# TODO: move these into layers/sequence.py
# import layers.sequence
# sequence.Recurrence(), sequence.Convolution()
# from layers import *  will import sequence.Recurrence()
# from layers.sequence import * will also import sequence.reduce(), but override non-seq Convolution.

# TODO: allow to say sequential=False, axis=2, length=100, ... something like this
def RecurrenceFrom(over_function, go_backwards=default_override_or(False), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that runs a cell function recurrently over a time sequence, with initial state.
    This form is meant for use in sequence-to-sequence scenarios.
    The difference to Recurrence() is that this returns a function that accepts the initial state as data argument(s).
    Initial state consists of N arguments, matching 'over'.
    '''

    go_backwards  = get_default_override(RecurrenceFrom, go_backwards=go_backwards)

    over_function = _sanitize_function(over_function)

    # get signature of cell
    *prev_state_args, _ = over_function.signature

    if len(over_function.outputs) != len(prev_state_args):
        # TODO: better say right here what the requirement is!
        raise TypeError('RecurrenceFrom: number of state variables inconsistent between create_placeholder() and recurrent block')

    # function that this layer represents
    def _recurrence_from_n(*args):
        *initial_state, x = args

        out_vars_fwd = [ForwardDeclaration(name=state_var.name) for state_var in prev_state_args] # create list of placeholders for the state variables

        # previous function; that is, past or future_value with initial_state baked in
        prev_out_vars = [delay(out_var, initial_state=init, time_step=-1 if go_backwards else +1) for out_var, init in zip(out_vars_fwd, initial_state)]  # delay (state vars)

        # apply the recurrent block ('over_function')
        out = over_function(*(prev_out_vars + [x]))  # over_function is a Function (previous outputs..., x) -> (state vars...)

        # connect the recurrent dependency
        for (var_fwd, var) in zip(out_vars_fwd, list(out.outputs)):
            var_fwd.resolve_to(var)

        if not return_full_state:
            out = combine([out.outputs[0]])  # BUGBUG: Without combine(), it fails with "RuntimeError: Runtime exception". Likely the known ref-counting bug. TODO: fix this inside Function(lambda)?

        return out

    # functions that this layer represents
    # The @Function pattern only supports fixed signatures, so we need one for each #states we support.
    # TODO: undo x_last hack
    def recurrence_from_1(h, x_last):
        return _recurrence_from_n(h, x_last)
    def recurrence_from_2(h, c, x_last):
        return _recurrence_from_n(h, c, x_last)
    def recurrence_from_3(h, c, a, x_last):
        return _recurrence_from_n(h, c, a, x_last)

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
    Layer factory function to create a function that runs a cell function recurrently over a time sequence.
    This form is meant for use in regular recurrent-model scenarios.
    ``initial_state`` must be a constant (or at least have known shape). To pass initial_state as a data input, use RecurrenceFrom() instead.
    TODO: Can bidirectionality be an option of this? bidirectional=True? What was the reason it cannot?
    '''

    go_backwards  = get_default_override(Recurrence, go_backwards=go_backwards)
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    over_function = _sanitize_function(over_function)

    # get signature of cell
    *prev_state_args, _ = over_function.signature

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
        return recurrence_from(*(initial_state + (x,)))

    recurrence = _inject_name(recurrence, name)

    return Block(recurrence, 'Recurrence', Record(over_function=over_function))


def Fold(folder_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that implements the fold() catamorphism.
    ``go_backwards=False`` selects a fold-left, while ``True`` a fold-right,
    but note that the ``folder_function`` signature is always the one of fold-left.
    Like ``Recurrence()`` but returns only the final state.
    '''

    go_backwards  = get_default_override(Fold, go_backwards=go_backwards)
    initial_state = get_default_override(Fold, initial_state=initial_state)

    # get the scan function
    recurrence = Recurrence(folder_function, go_backwards=go_backwards, initial_state=initial_state, return_full_state=return_full_state)

    # now take the last or first
    get_final = sequence.first if go_backwards else sequence.last
    fold = recurrence >> tuple(get_final for output in recurrence.outputs)

    fold = _inject_name(fold, name)

    return Block(fold, 'Fold', Record(folder_function=folder_function))


# TODO: This is still a bit messy. The returned unfold_from() function should take the encoding instead of 'input'.
def UnfoldFrom(generator_function, map_state_function=identity, until_predicate=None, length_increase=1, initial_state=None, name=''):
    '''
    Layer factory function to create a function that implements the unfold() anamorphism. It creates a function that, starting with a seed input,
    applies 'generator_function' repeatedly and emits the sequence of results. Depending on the recurrent block,
    it may have this form:
       `result = f(... f(f([g(input), initial_state])) ... )`
    or this form:
       `result = f(g(input), ... f(g(input), f(g(input), initial_state)) ... )`
    where `f` is `generator_function`.
    An example use of this is sequence-to-sequence decoding, where `g(input)` is the sequence encoder,
    `initial_state` is the sentence-start symbol, and `f` is the decoder. The first
    of the two forms above is a plain sequence-to-sequence model where encoder output
    is the start state for the output recursion.
    The second form is an attention-based decoder, where the encoded input affects every application
    of `f` differently.
    '''

    generator_function = _sanitize_function(generator_function)
    map_state_function = _sanitize_function(map_state_function)
    until_predicate    = _sanitize_function(until_predicate)

    # check the signature of the passed function
    if len(generator_function.signature) != 1 or len(generator_function.outputs) < 1 or len(generator_function.outputs) > 2:
        raise TypeError('generator_function should take 1 positional argument (state) and return a single output or a tuple (output, new state)')

    # TODO: having to pass the dynamic axis is suboptimal. Any better way?
    # BUGBUG: The name _from indicates that the start state should be passed.
    @Function
    def unfold_from(dynamic_axes_like):
        # create a new dynamic axis if a length increase is specified
        out_axis = dynamic_axes_like
        if length_increase != 1:
            factors = sequence.constant_with_dynamic_axes_like(length_increase, out_axis) # repeat each frame 'length_increase' times, on average
            out_axis = sequence.where(factors)  # note: values are irrelevant; only the newly created axis matters

        state_fwd = ForwardDeclaration()
        #state_fwd = placeholder_variable(name='ph')   #ForwardDeclaration()
        prev_state = delay(state_fwd, initial_state=initial_state)
        #prev_state = past_value(state_fwd, initial_state=initial_state)
        #print('gen args:', [arg.name for arg in generator_function.arguments])
        z = generator_function(prev_state) # returns either (output) or (output, new state)
        #from cntk.ops.functions import CloneMethod
        #z = generator_function.clone(CloneMethod.share, {generator_function.arguments[0]: prev_state,
        #                                                 generator_function.arguments[1]: generator_function.arguments[1]}) # returns either (output) or (output, new state)
        #print('z args:', [arg.name for arg in z.arguments])
        output = z.outputs[0]
        new_state = z.outputs[1] if len(z.outputs) > 1 else output # we allow generator to return a single value if it is identical to the new state
        #output = combine([z.outputs[0]])   # BUGBUG: ref-count issue
        #new_state = combine([z.outputs[1]]) if len(z.outputs) > 1 else output # we allow generator to return a single value if it is identical to the new state
        # apply map_state_function if given
        new_state = map_state_function(new_state)
        # implant the dynamic axis (from dynamic_axes_like)
        from ..utils import sanitize_input, typemap
        from ..cntk_py import reconcile_dynamic_axis
        new_state = typemap(reconcile_dynamic_axis)(sanitize_input(new_state), sanitize_input(out_axis))
        #new_state = combine([new_state])
        state_fwd.resolve_to(new_state)
        # BUGBUG: Could it be this?
        #new_state.output.owner.replace_placeholders({state_fwd: new_state.output})
        #new_state.replace_placeholders({state_fwd: new_state.output})
        #print('state_fwd after resolve:', [arg.name for arg in new_state.signature])
        #print('state_fwd after resolve, args:', [arg.name for arg in new_state.arguments])

        output = combine([output]) # BUGBUG: without this, it crashes with bad weak ptr
        # BUGBUG: MUST do this after resolving the recurrence, otherwise also crashes

        #print('output args 1:', [arg.name for arg in output.arguments])
        #state_fwd = None
        #print('output args 2:', [arg.name for arg in output.arguments])
        #prev_state = None
        #print('output args 3:', [arg.name for arg in output.arguments])
        #z = None
        #print('output args 4:', [arg.name for arg in output.arguments])
        #new_state = None
        #print('output args 5:', [arg.name for arg in output.arguments])

        # apply until_predicate if given
        if until_predicate is not None:
            valid_frames = Recurrence(lambda x, h: (1-past_value(x)) * h, initial_state=1)(until_predicate(output))
            output = sequence.gather(output, valid_frames)
        # BUGBUG: used to work, but now fails with "Node '__v2libuid__Slice28080__v2libname__Slice22511' (Slice operation): DataFor: FrameRange's dynamic axis is inconsistent with matrix: {numTimeSteps:1, numParallelSequences:1, sequences:[{seqId:0, s:0, begin:0, end:1}]} vs. {numTimeSteps:11, numParallelSequences:1, sequences:[{seqId:0, s:0, begin:0, end:11}]}"

        #print('output:', [arg.name for arg in output.signature])
        #print('output args:', [arg.name for arg in output.arguments])
        return output

    unfold_from = _inject_name(unfold_from, name)

    return Block(unfold_from, 'UnfoldFrom', Record(generator_function=generator_function))
