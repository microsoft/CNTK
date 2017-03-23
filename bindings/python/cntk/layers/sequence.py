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
    Delay(T=1, initial_state=0, name='')

    Layer factory function to create a layer that delays input the input by a given number of time steps. Negative means future.
    This is provided as a layer that wraps :func:`~cntk.ops.sequence.delay` so that it can easily be used in a Sequential() expression.

    Example:
        >>> # create example input: one sequence with 4 tensors of shape (3, 2)
        >>> from cntk.layers import Input, Sequential
        >>> from cntk.layers.typing import Tensor, Sequence
        >>> x = Input(**Sequence[Tensor[2]])
        >>> x0 = np.reshape(np.arange(6,dtype=np.float32),(1,3,2))
        >>> x0
        array([[[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]]], dtype=float32)
        >>> # trigram expansion: augment each item of the sequence with its left and right neighbor
        >>> make_trigram = Sequential([tuple(Delay(T) for T in (-1,0,1)),  # create 3 shifted versions
        ...                            splice])                            # concatenate them
        >>> y = make_trigram(x)
        >>> y(x0)
        array([[[ 2.,  3.,  0.,  1.,  0.,  0.],
                [ 4.,  5.,  2.,  3.,  0.,  1.],
                [ 0.,  0.,  4.,  5.,  2.,  3.]]], dtype=float32)
        >>> #    --(t-1)--  ---t---  --(t+1)--      

    Args:
        T (int): the number of time steps to look into the past, where negative values mean to look into the future, and 0 means a no-op (default 1).
        initial_state: tensor or scalar representing the initial value to be used when the input tensor is shifted in time.
        name (str, optional): the name of the Function instance in the network

    Returns:
        cntk.ops.functions.Function: 
        A function that accepts one argument (which must be a sequence) and returns it delayed by ``T`` steps
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
    PastValueWindow(window_size, axis, go_backwards=False, name='')

    Layer factory function to create a function that returns a static, maskable view for N past steps over a sequence along the given 'axis'.
    It returns two matrices: a value matrix, shape=(N,dim), and a valid window, shape=(N,1).

    This is used for attention modeling. CNTK presently does not support nested dynamic axes.
    Since attention models require nested axes (encoder hidden state vs. decoder hidden state),
    this layer can be used to map the encoder's dynamic axis to a static tensor axis.
    The static axis has a maximum length (``window_size``). To account for shorter input
    sequences, this function also returns a validity mask of the same axis dimension.
    Longer sequences will be truncated.

    Example:
        >>> # create example input: one sequence with 4 tensors of shape (3, 2)
        >>> from cntk.layers import Input, Sequential
        >>> from cntk.layers.typing import Tensor, Sequence
        >>> x = Input(**Sequence[Tensor[2]])
        >>> x0 = np.reshape(np.arange(6,dtype=np.float32),(1,3,2))
        >>> x0
        array([[[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]]], dtype=float32)
        >>> # convert dynamic-length sequence to a static-dimension tensor
        >>> to_static_axis = PastValueWindow(4, axis=-2)  # axis=-2 means second last
        >>> y = to_static_axis(x)
        >>> value, valid = y(x0)
        >>> # 'value' contains the items from the back, padded with 0
        >>> value
        array([[[ 4.,  5.],
                [ 2.,  3.],
                [ 0.,  1.],
                [ 0.,  0.]]], dtype=float32)
        >>> # 'valid' contains a scalar 1 for each valid item, and 0 for the padded ones
        >>> # E.g., when computing the attention softmax, only items with a 1 should be considered.
        >>> valid
        array([[[ 1.],
                [ 1.],
                [ 1.],
                [ 0.]]], dtype=float32)

    Args:
        window_size (int): maximum number of items in sequences. The `axis` will have this dimension.
        axis (int or :class:`~cntk.axis.Axis`, optional, keyword only): axis along which the
         concatenation will be performed
        name (str, optional, keyword only): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`: 
        A function that accepts one argument, which must be a sequence. It returns a fixed-size window of the last ``window_size`` items,
        spliced along ``axis``.
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
    Layer factory function to create a function that runs a cell function recurrently over an input sequence, with initial state.
    This layer is very similar to :func:`~cntk.layers.sequence.Recurrence`,
    except that the initial state is data dependent, and thus passed to the layer function as a data input
    rather than as an initialization parameter to the factory function.
    This form is meant for use in sequence-to-sequence scenarios.
    This documentation only covers this case; for additional information on parameters, see :func:`~cntk.layers.sequence.Recurrence`.

    The layer function returned by this factory function accepts the initial state as data argument(s).
    It has the signature ``(initial_state, input_sequence) -> output_sequence``.
    If the step function has multiple state variables, then the first N parameters are the initial state variables.

    The initial state can be non-sequential data, as one would have for a plain sequence-to-sequence model,
    or sequential data. In the latter case, the last item is the initial state.

    Example:
     >>> from cntk.layers import *
     >>> from cntk.layers.typing import *

     >>> # a plain sequence-to-sequence model in training (where label length is known)
     >>> en = Input(**SequenceOver[Axis('m')][SparseTensor[20000]])  # English input sentence
     >>> fr = Input(**SequenceOver[Axis('n')][SparseTensor[30000]])  # French target sentence

     >>> embed = Embedding(300)
     >>> encoder = Recurrence(LSTM(500), return_full_state=True)
     >>> decoder = RecurrenceFrom(LSTM(500))       # decoder starts from a data-dependent initial state, hence -From()
     >>> emit = Dense(30000)
     >>> h, c = encoder(embed(en)).outputs         # LSTM encoder has two outputs (h, c)
     >>> z = emit(decoder(h, c, past_value(fr)))   # decoder takes encoder outputs as initial state
     >>> loss = C.cross_entropy_with_softmax(z, fr)

    Args:
     step_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      This function must have N+1 inputs and N outputs, where N is the number of state variables
      (typically 1 for GRU and plain RNNs, and 2 for LSTMs).
     go_backwards (bool, defaults to ``False``): if ``True`` then run the recurrence from the end of the sequence to the start.
     initial_state (scalar or tensor without batch dimension; or a tuple thereof):
      the initial value for the state. This can be a constant or a learnable parameter.
      In the latter case, if the step function has more than 1 state variable,
      this parameter must be a tuple providing one initial state for every state variable.
     return_full_state (bool, defaults to ``False``): if ``True`` and the step function has more than one
      state variable, then the layer returns a all state variables (a tuple of sequences);
      whereas if not given or ``False``, only the first state variable is returned to the caller.
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`: 
        A function that accepts arguments ``(initial_state_1, initial_state_2, ..., input_sequence)``,
        where the number of initial state variables must match the step function's.
        The initial state can be a sequence, in which case its last (or first if ``go_backwards``) item is used.
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


# TODO: Can bidirectionality be an option of this? bidirectional=True?
def Recurrence(step_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name=''):
    '''
    Layer factory function to create a function that runs a step function recurrently over an input sequence.
    This implements the typical recurrent model.

    The step function can be any :class:`~cntk.ops.functions.Function` or Python function
    with a signature ``(h_prev, x) -> h``, where ``h_prev`` is the previous state, ``x`` is the new
    data input, and the output is the new state.
    All three are sequences of the same length. The step function will be called item by item.

    Step functions can have more than one state output, e.g. :func:`~cntk.layers.blocks.LSTM`.
    In this case, the first N arguments are the previous state, followed by one more argument that
    is the data input; and its output must be a tuple of N values.
    In this case, the recurrence operation will, by default, return the first of the state variables
    (in the LSTM case, the ``h``), while additional state variables are internal (like the LSTM's ``c``).
    If all state variables should be returned, pass ``return_full_state=True``.

    Typical step functions are :func:`~cntk.layers.blocks.LSTM`, :func:`~cntk.layers.blocks.GRU`, and :func:`~cntk.layers.blocks.RNNUnit`.
    However, any function with a signature as described above is admissible.
    For example, a cumulative sum over a sequence can be computed as ``Recurrence(plus)``,
    or a GRU layer with projection could be realized as ``Recurrence(GRU(500) >> Dense(200))``;
    where the projection is applied to the hidden state as fed back to the next step.
    ``F>>G`` is a short-hand for ``Sequential([F, G])``.

    Optionally, the recurrence can run backwards. This is useful for constructing bidirectional models.

    ``initial_state`` must be a constant. To pass initial_state as a data input, e.g. for a sequence-to-sequence
    model, use :func:`~cntk.layers.sequence.RecurrenceFrom()` instead.

    Note: ``Recurrence()`` is the equivalent to what in functional programming is often called ``scanl()``.

    Example:
     >>> from cntk.layers import Input, Constant, Sequential
     >>> from cntk.layers.typing import Tensor, Sequence

     >>> # a recurrent LSTM layer
     >>> lstm_layer = Recurrence(LSTM(500))

     >>> # a bidirectional LSTM layer
     >>> # using function tuples to implement a bidirectional LSTM
     >>> bi_lstm_layer = Sequential([(Recurrence(LSTM(250)),                      # first tuple entry: forward pass
     ...                              Recurrence(LSTM(250), go_backwards=True)),  # second: backward pass
     ...                             splice])                                     # splice both on top of each other
     >>> bi_lstm_layer.update_signature(Sequence[Tensor[13]])
     >>> bi_lstm_layer.shape   # shape reflects concatenation of both output states
     (500,)
     >>> tuple(str(axis.name) for axis in bi_lstm_layer.dynamic_axes)  # (note: str() needed only for Python 2.7)
     ('defaultBatchAxis', 'defaultDynamicAxis')

     >>> # cumulative sum over inputs
     >>> x = Input(**Sequence[Tensor[2]])
     >>> x0 = np.array([[   3,    2],
     ...                [  13,   42],
     ...                [-100, +100]])
     >>> cum_sum = Recurrence(C.plus, initial_state=Constant([0, 0.5]))
     >>> y = cum_sum(x)
     >>> y(x0)
     array([[[   3. ,    2.5],
             [  16. ,   44.5],
             [ -84. ,  144.5]]], dtype=float32)

    Args:
     step_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      This function must have N+1 inputs and N outputs, where N is the number of state variables
      (typically 1 for GRU and plain RNNs, and 2 for LSTMs).
     go_backwards (bool, defaults to ``False``): if ``True`` then run the recurrence from the end of the sequence to the start.
     initial_state (scalar or tensor without batch dimension; or a tuple thereof):
      the initial value for the state. This can be a constant or a learnable parameter.
      In the latter case, if the step function has more than 1 state variable,
      this parameter must be a tuple providing one initial state for every state variable.
     return_full_state (bool, defaults to ``False``): if ``True`` and the step function has more than one
      state variable, then the layer returns a all state variables (a tuple of sequences);
      whereas if not given or ``False``, only the first state variable is returned to the caller.
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`: 
        A function that accepts one argument (which must be a sequence) and performs the recurrent operation on it
    '''

    # BUGBUG: the cum_sum expression in the docstring should be this:
    #cum_sum = Recurrence(C.plus, initial_state=np.array([0, 0.5]))
    # BUGBUG: whereas passing a NumPy array fails with "TypeError: cannot convert value of dictionary"
    #cum_sum = Recurrence(C.plus, initial_state=Constant([0, 0.5]))

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
    Fold(folder_function, go_backwards=False, initial_state=0, return_full_state=False, name='')

    Layer factory function to create a function that runs a step function recurrently over an input sequence,
    and returns the final state.
    This is often used for embeddings of sequences, e.g. in a sequence-to-sequence model.

    ``Fold()`` is the same as :func:`~cntk.layers.sequence.Recurrence` except that only the final state is returned
    (whereas ``Recurrence()`` returns the entire state sequence).
    Hence, this documentation will only focus on the differences to ``Recurrence()``,
    please see :func:`~cntk.layers.sequence.Recurrence` for a detailed information on parameters.

    Commonly, the ``folder_function`` is a recurrent block such as an LSTM.
    However, one can pass any binary function. E.g. passing ``plus`` will sum
    up all items of a sequence; while ``element_max`` would perform a max-pooling over all items of the sequence.

    Note: CNTK's Fold() is similar to the fold() catamorphism known from functional programming.
    ``go_backwards=False`` corresponds to a fold-left, and ``True`` to a fold-right,
    except that the ``folder_function`` signature is always the one of fold-left.

    Example:
     >>> from cntk.layers import *
     >>> from cntk.layers.typing import *

     >>> # sequence classifier. Maps a one-hot word sequence to a scalar probability value.
     >>> # The recurrence is a Fold(), meaning only the final hidden state is produced.
     >>> # The Label() layer allows to access the final hidden layer by name.
     >>> sequence_classifier = Sequential([ Embedding(300),
     ...                                    Fold(LSTM(500)),
     ...                                    Dense(1, activation=sigmoid) ])

     >>> # element-wise max-pooling over an input sequence
     >>> x = Input(**Sequence[Tensor[2]])
     >>> x0 = np.array([[ 1, 2 ],
     ...                [ 6, 3 ],
     ...                [ 4, 2 ],
     ...                [ 8, 1 ],
     ...                [ 6, 0 ]])
     >>> seq_max_pool = Fold(C.element_max)
     >>> y = seq_max_pool(x)
     >>> y(x0)
         array([[ 8.,   3.]], dtype=float32)

     >>> # element-wise sum over an input sequence
     >>> seq_sum = Fold(C.plus)
     >>> y = seq_sum(x)
     >>> y(x0)
         array([[ 25.,   8.]], dtype=float32)

    Args:
     folder_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      This function must have N+1 inputs and N outputs, where N is the number of state variables
      (typically 1 for GRU and plain RNNs, and 2 for LSTMs).
     go_backwards (bool, defaults to ``False``): if ``True`` then run the recurrence from the end of the sequence to the start.
     initial_state (scalar or tensor without batch dimension; or a tuple thereof):
      the initial value for the state. This can be a constant or a learnable parameter.
      In the latter case, if the step function has more than 1 state variable,
      this parameter must be a tuple providing one initial state for every state variable.
     return_full_state (bool, defaults to ``False``): if ``True`` and the step function has more than one
      state variable, then the layer returns the final value of a all state variables (a tuple of sequences);
      whereas if not given or ``False``, only the final value of the first of the state variables is returned to the caller.
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`: 
        A function that accepts one argument (which must be a sequence) and performs the fold operation on it
    '''

    go_backwards  = get_default_override(Fold, go_backwards=go_backwards)
    initial_state = get_default_override(Fold, initial_state=initial_state)

    # get the scan function
    recurrence = Recurrence(folder_function, go_backwards=go_backwards, initial_state=initial_state, return_full_state=return_full_state)

    # now take the last or first
    get_final = sequence.first if go_backwards else sequence.last
    fold = recurrence >> tuple(get_final for output in recurrence.outputs)

    return _inject_name(fold, name)


# TODO: This API is still suboptimal, and should be fixed as follows:
#  - the returned layer function should take the initial_state
#  - the input length factor should be a layer function; in addition, a fixed max length should be possible
#  - map_state_function is unused and should be removed
#  - BUGBUG: tuple-valued state should be supported
def UnfoldFrom(generator_function, map_state_function=identity, until_predicate=None, length_increase=1, initial_state=None, name=''):
    '''
    UnfoldFrom(generator_function, until_predicate=None, length_increase=1, initial_state=None, name='')

    Layer factory function to create a function that implements a recurrent generator.
    Starting with a seed state, the ``UnfoldFrom()`` layer
    repeatedly applies ``generator_function`` and emits the sequence of results.
    ``UnfoldFrom(f, initial_state=s)``
    emits the sequence ``f(s), f(f(s)), f(f(f(s))), ...``.
    ``s`` can be tuple-valued.

    A typical application is the decoder of a sequence-to-sequence model,
    the generator function ``f`` accepts a two-valued state, with the first
    being an emitted word, and the second being an internal recurrent state.
    The initial state would be a tuple ``(w0, h0)``
    where ``w0`` represents the sentence-start symbol,
    and ``h0`` is a thought vector that encodes the input sequence (as obtained
    from a :func:`~cntk.layers.sequence.Fold()` operation).

    A variant allows the state and the emitted sequence to be different. In that case,
    ``f`` returns a tuple (output value, new state), and
    ``UnfoldFrom(f, initial_state=s)``
    would emit the sequence ``f(s)[0], f(f(s)[1])[0], f(f(f(s)[1])[1])[0], ...``.

    The maximum length of the output sequence is not unlimited, but determined by the argument to
    the layer function, multiplied by an optional increase factor.

    Optionally, a function can be provided to denote that the end of the sequence has been reached.

    Note: In the context of functional programming, the first form of this operation is known as the unfold() anamorphism.

    Example:
     TO BE PROVIDED after signature changes.

    Args:
     generator_function (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      This function must have N inputs and a N-tuple-valued output, where N is the number of state variables.
      If the emitted value should be different from the state, then the function should have
      a tuple of N+1 outputs, where the first output is the value to emit, while the others are the state.
     until_predicate (:class:`~cntk.ops.functions.Function` or equivalent Python function):
      A function that denotes when the last element of the unfold has been emitted.
      It takes the same number of argments as the generator, and returns a scalar that must be 1
      for the last element of the sequence, and 0 otherwise.
      This is subject to the maximum length as determined by the input sequence and ``length_increase``.
      If this parameter is not provided, the output length will be equal to the specified maximum length.
     length_increase (float, defaults to 1): the maximum number of output items is equal to the
      number of items of the argument to the unfold function, multiplied by this factor.
      For example, pass 1.5 here if the output sequence can be at most 50% longer than the input.
     initial_state (scalar or tensor without batch dimension; or a tuple thereof):
      the seed value for the state
     name (str, optional): the name of the Function instance in the network

    Returns:
        :class:`~cntk.ops.functions.Function`: 
        A function that accepts one argument (which must be a sequence and provides
        a reference for the maximum length of the output sequence), and performs the unfold operation on it
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
