# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Higher-order functions, like :func:`Sequential` and :func:`ResNetBlock`. Note that
sequential higher-order functions like :func:`~cntk.layers.sequence.Recurrence` are in :mod:`cntk.layers.sequence`.
'''

from types import FunctionType
from inspect import getargspec

from ..variables import Record
from .blocks import *
from .blocks import _initializer_for, _get_initial_state_or_default, _INFERRED, _inject_name
from .sequence import * # they are also higher-order functions
from .typing import *

# TODO: should we have a parameter to specify the arity of the input?
#       Can it be automatically determined? (yes, unless the first function is a tuple, then we don't know whether to broadcast or not)
def Sequential(layers, name=''):
    '''
    Sequential(layers, name='')

    Layer factory function to create a composite that applies a sequence of layers (or any functions) onto an input.
    ``Sequential ([F, G, H])(x)`` means the same as ``H(G(F(x)))``.

    The list of functions may also include tuples of functions. In that case, each function
    in a tuple is applied to the input, and the result is a tuple containing the results of
    these function applications. If followed by another function (typ. ``plus`` or ``splice``),
    the tuple items form the arguments to that function.

    Intermediate values in the chain can be accessed by name by inserting a ``Label(name=...)`` layer.

    Note: An equivalent way of writing ``Sequential ([F, G, H])(x)`` is ``F >> G >> H``.

    Example:
     >>> from cntk.layers import *

     >>> # sequence classifier. Maps a one-hot word sequence to a scalar probability value.
     >>> # The recurrence is a Fold(), meaning only the final hidden state is produced.
     >>> # The Label() layer allows to access the final hidden layer by name.
     >>> model = Sequential([Embedding(300), Fold(LSTM(500)), Label('hidden'), Dense(1, activation=sigmoid)])
     >>> model.update_signature(Sequence[Tensor[30000]])
     >>> model.hidden.shape
         (500,)

     >>> # simple example that squares an input value
     >>> f = Sequential([log, lambda x: 2 * x, exp])  # the second function is a Python lambda
     >>> f.update_signature(1)
     >>> f([np.array([2])])     # log, times 2, exp is the same as computing the square
         array([[ 4.]], dtype=float32)

     >>> # using function tuples to implement a bidirectional LSTM
     >>> bi_lstm = Sequential([(Recurrence(LSTM(250)),                      # first tuple entry: forward pass
     ...                        Recurrence(LSTM(250), go_backwards=True)),  # second: backward pass
     ...                       splice])                                     # splice both on top of each other

     >>> # using function tuple to implement a ResNet block
     >>> # The function tuple applies all items to the input, and emits a tuple with the results
     >>> # that then act as the arguments to the next one.
     >>> # Here we say (Convolution(), identity), which generates two arguments to the next function,
     >>> # the first being the convolution, the second being the input passed through.
     >>> # Following that with plus() implements the ResNet formula.
     >>> from cntk.ops import plus, relu
     >>> resnet_layer = Sequential([(Convolution((3,3), 64, activation=None), # first tuple entry
     ...                             identity),                               # second tuple entry is a pass-through
     ...                            plus,                                     # this sums both
     ...                            relu])                                    # activation applied afterwards

     >>> # simple function-tuples example with values
     >>> f = Sequential([(lambda x: x * x, identity), splice])  # computes tuple (x^2, x) and splices both values
     >>> f.update_signature(1)
     >>> f([np.array([2])])
         array([[ 4.,  2.]], dtype=float32)

    Args:
      layers (list of :class:`~cntk.ops.functions.Function`, equivalent Python functions, tuples of functions, or lists thereof): the list of functions to apply in sequence.
        A tuple aplies each of its items to the input and results in a tuple value.
        An item that is a list will be flattened.

    Returns:
        cntk.ops.functions.Function: 
        A function that accepts one argument and applies the given ``functions`` one after another.
    '''
    if not isinstance(layers, list): # to support nested lists, run every item recursively through Sequential()
        # TODO: Is this confusing w.r.t. tuple which is parallel and list which is sequential?
        return layers
    from functools import reduce
    layers = [Sequential(layer) for layer in layers] # expand all layers recursively
    composed_function = reduce(lambda f, g: f >> g, layers, identity)

    return _inject_name(composed_function, name)


def For(what_range, constructor, name=''):
    '''
    For(what_range, constructor, name='')

    Layer factory function to create a composite through a pattern similar to Python's `for` statement.

    This layer factory loops over the given range and passes each value to the constructor function.
    It is equivalent to
    ``Sequential([constructor(i) for i in what_range])``.

    It is acceptable that ``constructor`` takes no argument.

    Example:
     >>> from cntk.layers import *
     >>> from cntk.ops import relu

     >>> # stack of 3 Dense relu layers
     >>> model = For(range(3), lambda: Dense(2000, activation=relu))

     >>> # version of the above that has no activation for the last layer
     >>> model = For(range(3), lambda i: Dense(2000, activation=relu if i < 2 else identity))

     >>> # complex example that uses For() inside Sequential()
     >>> with default_options(activation=relu, pad=True):  # default activation is relu
     ...     model = Sequential([
     ...          For(range(2), lambda : [
     ...              Convolution2D((3,3), 64), 
     ...              Convolution2D((3,3), 64), 
     ...              MaxPooling((3,3), strides=2)
     ...          ]), 
     ...          Label('ndfeat'),              # name this specific value
     ...          For(range(2), lambda i: [     # this passes a nested list to Sequential
     ...              Dense([256,128][i]),      # layer index i used to index into an array of parameters
     ...              Dropout(0.5)
     ...          ]), 
     ...          Label('hidden'),
     ...          Dense(10, activation=None)    # activation parameter overrides default (which was set to relu)
     ...      ])
     >>> model.update_signature((3,32,32))      # RGB, 32 x 32 pixels
     >>> model.ndfeat.shape                     # shape at top of convo/pooling pyramid
         (64, 8, 8)
     >>> model.hidden.shape                     # shape before classifier
         (128,)

    Args:
     what_range (range): a Python range to loop over
     constructor (Python function/lambda with 1 or 0 arguments): lambda that constructs a layer

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the layers as constructed by ``constructor`` one after another.
    '''
    # Python 2.7 support requires us to use getargspec() instead of inspect
    takes_arg = len(getargspec(constructor).args) > 0

    # For Python 3, check if it is a python function/lambda
    if type(constructor) != FunctionType or not callable(constructor):
        raise ValueError("constructor must be a Python function/lambda")

    # helper to call the layer constructor
    def call(i):
        if takes_arg:
            return constructor(i)  # takes an arg: pass it
        else:
            return constructor()   # takes no arg: call without, that's fine too

    layers = [call(i) for i in what_range]
    sequential = Sequential(layers)

    return _inject_name(sequential, name)


# legacy name for For()
def LayerStack(N, constructor):
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
            'For(...) instead', DeprecationWarning)
    return For(range(N), constructor)


def SequentialClique(functions, name=''):
    '''
    SequentialClique(functions, name='')

    Layer factory function to create a composite that applies a sequence of functions onto an input,
    with skip connections between all function. I.e. each function receives a sum of the input and all
    prior functions' outputs.

    Example:
     >>> from cntk.layers import *
     >>> from cntk.ops import abs, sqrt, square
     >>> x = input(2)
     >>> seq_clique = SequentialClique([abs, sqrt, square])
     >>> seq_clique(x).eval(np.array([2, 8], np.float32)) # 400 = square((8 + abs(8)) + sqrt(8 + abs(8)))
         array([[  36.,  400.]], dtype=float32)

    Args:
     functions (single or list of :class:`~cntk.ops.functions.Function`): functions to be applied.

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the sequence of functions.
    '''
    def clique(x):
        for f in functions:
            out = f(x)
            # BUGBUG: this should be a splice(), and it should be along depth.
            #         Interface to be finalized.
            x = x + out
        return out

    clique = _inject_name(clique, name)

    return clique


# TODO: consider potential name clash; users might want to call their functions the same.
def ResNetBlock(f, name=''):
    '''
    ResNetBlock(f, name='')

    Layer factory function to create a composite that adds a skip connection to a function.
    This is equivalent to ``Sequential((f, identity), plus)``.

    Example:
     >>> # a ResNet layer
     >>> from cntk.layers import *
     >>> from cntk.ops import relu
     >>> resnet_layer = Sequential([ResNetBlock(Convolution((3,3), 64, activation=None)), relu])

    Args:
      f (:class:`~cntk.ops.functions.Function` or equivalent Python function):
       the function to add the skip connection to.

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument, applies ``f`` to it, and adds the original argument.
    '''
    def skip(x):
        return f(x) + x

    skip = _inject_name(skip, name)

    return skip
