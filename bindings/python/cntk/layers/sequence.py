# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# sequence -- first/higher-order functions over sequences, like Recurrence()

from ..utils import Record
from ..ops import combine, past_value, future_value, splice, sequence
from .blocks import *
from .blocks import _get_initial_state_or_default, _inject_name


def Delay(T=1, initial_state=default_override_or(0), name=''):
    '''
    Layer factory function to create a layer that delays input the input by a given number of time steps. Negative means future.
    This is provided as a layer instead of a function so that it can easily be used in a Sequential() expression.
    '''
    initial_state = get_default_override(Delay, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    # expression
    @BlockFunction('Delay', name)
    def delay(x):
        # TODO: reenable this
        ## if specific dynamic_axes requested then delay without and inject a reconcile_dynamic_axis() on top
        #if dynamic_axes_like:
        #    r = delay(x, initial_state=initial_state, time_step=time_step, name='')
        #    from .utils import sanitize_input, typemap
        #    from _cntk_py import reconcile_dynamic_axis
        #    r = typemap(reconcile_dynamic_axis)(sanitize_input(r), sanitize_input(dynamic_axes_like), name=name)
        #    return r;
        ## regular case
        return sequence.delay(x, initial_state=initial_state, time_step=T)
    return delay


# TODO: reconsider the name. Windowed()?
def PastValueWindow(window_size, axis, go_backwards=default_override_or(False), name=''):
    '''
    Layer factory function to create a function that returns a static, maskable view for N past steps over a sequence along the given 'axis'.
    It returns two matrices: a value matrix, shape=(N,dim), and a valid window, shape=(1,dim)
    '''

    go_backwards = get_default_override(PastValueWindow, go_backwards=go_backwards)

    # helper to get the nth element
    def nth(input, offset):
        if go_backwards:
            final_f = sequence.first
            offset = -offset
        else:
            final_f = sequence.last
        return final_f(Delay(offset)(input))

    @BlockFunction('PastValueWindow', name)
    def past_value_window(x):
    
        ones_like_input = sequence.broadcast_as(1, x)

        # get the respective n-th element from the end
        last_values = [nth(x, t)               for t in range(window_size)]
        last_valids = [nth(ones_like_input, t) for t in range(window_size)]
    
        # stack rows 'beside' each other in a new static axis (create a new static axis that doesn't exist)
        value = splice(*last_values, axis=axis, name='value')
        valid = splice(*last_valids, axis=axis, name='valid')
    
        # value[t] = value of t steps back; valid[t] = true if there was a value t steps back
        return (value, valid)
    return past_value_window


def _sanitize_function(f):
    '''
    Helper to type-cast a Python function into a CNTK Function if not yet.
    '''
    import types
    if isinstance(f, types.FunctionType):
        f = Function(f)
    return f


# TODO: allow to say sequential=False, axis=2, length=100, ... something like this
def RecurrenceFrom(step_function, go_backwards=default_override_or(False), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that runs a cell function recurrently over a time sequence, with initial state.
    This form is meant for use in sequence-to-sequence scenarios.
    The difference to Recurrence() is that this returns a function that accepts the initial state as data argument(s).
    Initial state consists of N arguments, matching 'over'.
    '''

    go_backwards  = get_default_override(RecurrenceFrom, go_backwards=go_backwards)

    step_function = _sanitize_function(step_function)

    # get signature of cell
    #*prev_state_args, _ = step_function.signature  # Python 3
    prev_state_args = step_function.signature[0:-1]

    if len(step_function.outputs) != len(prev_state_args):
        # TODO: better say right here what the requirement is!
        raise TypeError('RecurrenceFrom: number of state variables inconsistent between create_placeholder() and recurrent block')

    # function that this layer represents
    def _recurrence_from_n(*args):
        #*initial_state, x = args # Python 3
        initial_state = args[:-1]
        x             = args[-1]

        out_vars_fwd = [ForwardDeclaration(name=state_var.name) for state_var in prev_state_args] # create list of placeholders for the state variables

        # previous function; that is, past or future_value with initial_state baked in
        if len(out_vars_fwd) != len(initial_state):
            raise ValueError('RecurrenceFrom() length mismatch between out_vars_fwd and initial_state. Should not happen')
        prev_out_vars = [sequence.delay(out_var, initial_state=init, time_step=-1 if go_backwards else +1) for out_var, init in zip(out_vars_fwd, initial_state)]  # delay (state vars)

        # apply the recurrent block ('step_function')
        out = step_function(*(prev_out_vars + [x]))  # step_function is a Function (previous outputs..., x) -> (state vars...)

        # connect the recurrent dependency
        for (var_fwd, var) in zip(out_vars_fwd, list(out.outputs)):
            var_fwd.resolve_to(var)

        if not return_full_state:
            out = combine([out.outputs[0]])  # BUGBUG: Without combine(), it fails with "RuntimeError: Runtime exception". Likely the known ref-counting bug.

        return out

    # functions that this layer represents
    # The @Function pattern only supports fixed signatures, so we need one for each #states we support.
    def recurrence_from_1(h, x):
        return _recurrence_from_n(h, x)
    def recurrence_from_2(h, c, x):
        return _recurrence_from_n(h, c, x)
    def recurrence_from_3(h, c, a, x):
        return _recurrence_from_n(h, c, a, x)

    recurrence_from_functions = [recurrence_from_1, recurrence_from_2, recurrence_from_3]
    num_state_args = len(prev_state_args)
    if num_state_args == 0 or num_state_args > len(recurrence_from_functions):
        raise ValueError('RecurrenceFrom() currently supports recurrence with up to {} state variables'.format(len(recurrence_from_functions)))

    # this creates the CNTK Function
    recurrence_from = Function(recurrence_from_functions[num_state_args-1])

    return _inject_name(recurrence_from, name)


def Recurrence(step_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that runs a cell function recurrently over a time sequence.
    This form is meant for use in regular recurrent-model scenarios.
    ``initial_state`` must be a constant (or at least have known shape). To pass initial_state as a data input, use RecurrenceFrom() instead.
    TODO: Can bidirectionality be an option of this? bidirectional=True? What was the reason it cannot?
    '''

    go_backwards  = get_default_override(Recurrence, go_backwards=go_backwards)
    initial_state = get_default_override(Recurrence, initial_state=initial_state)
    initial_state = _get_initial_state_or_default(initial_state)

    step_function = _sanitize_function(step_function)

    # get signature of cell
    #*prev_state_args, _ = step_function.signature  # Python 3
    prev_state_args = step_function.signature[0:-1]

    if len(step_function.outputs) != len(prev_state_args):
        raise TypeError('Recurrence: number of state variables inconsistent between create_placeholder() and recurrent block')

    # initial state can be a single value or one per state variable (if more than one, like for LSTM)
    if isinstance(initial_state, tuple) and len(initial_state) == 1:
        initial_state = initial_state[0]
    if not isinstance(initial_state, tuple):
        # TODO: if initial_state is a CNTK Function rather than an initializer, then require to pass it multiple times; otherwise broadcast to all
        initial_state = tuple(initial_state for out_var in prev_state_args)

    # express it w.r.t. RecurrenceFrom
    recurrence_from = RecurrenceFrom(step_function, go_backwards, return_full_state) # :: (x, state seq) -> (new state seq)

    # function that this layer represents
    @Function
    def recurrence(x):
        return recurrence_from(*(initial_state + (x,)))
    return _inject_name(recurrence, name)


def Fold(folder_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that implements the fold() catamorphism.
    It is like ``Recurrence()`` except that it returns only the final state.
    This is often used for embeddings of sequences, e.g. in a sequence-to-sequence model.

    Commonly, the ``folder_function`` is a recurrent block such as an LSTM.
    One can, however, pass any binary function. E.g. passing ``plus`` will sum
    up all items of a sequence; while ``element_max`` would perform a max-pooling over all items of the sequence.

    ``go_backwards=False`` selects a fold-left, while ``True`` a fold-right,
    but note that the ``folder_function`` signature is always the one of fold-left.
    '''

    go_backwards  = get_default_override(Fold, go_backwards=go_backwards)
    initial_state = get_default_override(Fold, initial_state=initial_state)

    # get the scan function
    recurrence = Recurrence(folder_function, go_backwards=go_backwards, initial_state=initial_state, return_full_state=return_full_state)

    # now take the last or first
    get_final = sequence.first if go_backwards else sequence.last
    fold = recurrence >> tuple(get_final for output in recurrence.outputs)

    return _inject_name(fold, name)


# TODO: This is still a bit messy. The returned unfold_from() function should take the encoding instead of 'input'.
def UnfoldFrom(generator_function, map_state_function=identity, until_predicate=None, length_increase=1, initial_state=None, name=''):
    '''
    Layer factory function to create a function that implements the unfold() anamorphism. It creates a function that, starting with a seed input,
    applies 'generator_function' repeatedly and emits the sequence of results. Depending on the recurrent block,
    it may have this form::

      result = f(... f(f([g(input), initial_state])) ... )

    or this form::

      result = f(g(input), ... f(g(input), f(g(input), initial_state)) ... )

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
    # BUGBUG: initial_state must be passed to unfold_from
    # We can still pass dynamic_axes_like; reads like "unfold from XXX along axis of YYY".
    # And if we can close over 'input' in the generator, we can also bake it into what we pass, i.e. the length.
    @Function
    def unfold_from(dynamic_axes_like):
    #def unfold_from(initial_state, dynamic_axes_like):
        # create a new dynamic axis if a length increase is specified
        out_axis = dynamic_axes_like
        if length_increase != 1:
            factors = sequence.broadcast_as(length_increase, out_axis) # repeat each frame 'length_increase' times, on average
            out_axis = sequence.where(factors)  # note: values are irrelevant; only the newly created axis matters

        state_fwd = ForwardDeclaration(name='unfold_state_fwd')
        prev_state = sequence.delay(state_fwd, initial_state=initial_state, name='unfold_prev_state')
        # TODO: must allow multiple variables, just like recurrence, as to allow beam decoding (permutation matrix)
        z = generator_function(prev_state) # returns either (output) or (output, new state)
        output = z.outputs[0]
        new_state = z.outputs[1] if len(z.outputs) > 1 else output # we allow generator to return a single value if it is identical to the new state
        # apply map_state_function if given
        new_state = map_state_function(new_state)
        # implant the dynamic axis (from dynamic_axes_like)
        from cntk.internal import sanitize_input, typemap
        from ..cntk_py import reconcile_dynamic_axis
        new_state = typemap(reconcile_dynamic_axis)(sanitize_input(new_state), sanitize_input(out_axis))
        new_state = combine([new_state], name='unfold_new_state')
        state_fwd.resolve_to(new_state)

        output = combine([output], name='unfold_output') # BUGBUG: without this, it crashes with bad weak ptr
        # BUGBUG: MUST do this after resolving the recurrence, otherwise also crashes

        # apply until_predicate if given
        if until_predicate is not None:
            valid_frames = Recurrence(lambda h, x: (1-past_value(x)) * h, initial_state=1, name='valid_frames')(until_predicate(output))
            output = sequence.gather(output, valid_frames, name='valid_output')

        return output

    return _inject_name(unfold_from, name)
