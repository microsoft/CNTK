# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
The CNTK typing module contains basic CNTK type meta-classes for :func:`~cntk.functions.Function.update_signature` and type signatures for the CNTK :class:`~cntk.ops.functions.Function` decorator.

The type of a CNTK :class:`~cntk.variables.Variable` is defined by five properties: `shape`, `dynamic_axes`, `is_sparse`, `dtype`, and `needs_gradient`.
Some API functions accept these variables as independent arguments, e.g. :class:`~cntk.input`.
The typing module provides a Pythonic way to represent the variable type properties as a single data object.

Python type syntax can be used to create such a record for the three main properties, `shape`, `dynamic_axes`, and `is_sparse`,
using :class:`~cntk.layers.typing.Tensor`,  :class:`~cntk.layers.typing.SparseTensor`,  :class:`~cntk.layers.typing.ParameterTensor`,
:class:`~cntk.layers.typing.Sequence`,  and :class:`~cntk.layers.typing.SequenceOver`.

Note: This new type system may undergo changes. Please give us feedback on github or stackoverflow

Example:
    >>> # Tensor[...] denotes a data variable (with implied batch dimension)
    >>> from cntk.layers.typing import *
    >>> tp = Tensor[13,42]
    >>> tp.shape
    (13, 42)
    >>> tp.is_sparse
    False
    >>> [str(axis.name) for axis in tp.dynamic_axes]
    ['defaultBatchAxis']

    >>> # SparseTensor[...] is a sparse Tensor
    >>> tp = SparseTensor[9000]
    >>> tp.is_sparse
    True

    >>> # if the first argument is np.float32 or np.float64 or np.float16, then this sets the dtype
    >>> tp = Tensor[np.float32,13,42]
    >>> print(tp.dtype == np.float32)
    True

This record can be directly passed to update_signature().

Example:
    >>> from cntk.layers import *
    >>> f = Dense(500)
    >>> f.update_signature(Tensor[13,42])
    >>> f.shape
    (500,)

    >>> # This is just the same as saying
    >>> f = Dense(500)
    >>> _ = f.replace_placeholders({f.arguments[0]: C.input_variable(shape=(13,42), dynamic_axes=[Axis.default_batch_axis()])})
    >>> f.shape
    (500,)

Scalars can be just specified as float or np.float32 or np.float64 or np.float16.

Example:
    >>> @Function
    ... def f(x):
    ...    return x+1
    >>> f.update_signature(np.float64)
    >>> print(f.dtype == np.float64)
    True

To specify types with a dynamic axis, use `Sequence[]`.

Example:
    >>> tp = Sequence[SparseTensor[9000]]
    >>> [str(axis.name) for axis in tp.dynamic_axes]
    ['defaultBatchAxis', 'defaultDynamicAxis']

This will refer to the default dynamic axis. If your model uses multiple dynamic axes, such as a sequence-to-sequence model,
you use `SequenceOver[]` to define your own sequence type for each.

Example:
    >>> InputSequence = SequenceOver[Axis('input')]
    >>> tp = InputSequence[SparseTensor[9000]]
    >>> [str(axis.name) for axis in tp.dynamic_axes]
    ['defaultBatchAxis', 'input']

The typing syntax can be used to directly define CNTK functions with their input types.
This is often done for the criterion function.

Example:
    >>> from cntk import debugging, cross_entropy_with_softmax
    >>> model = Sequential([Embedding(300), Fold(GRU(128)), Dense(10)])
    >>> print(model)
    Composite(keep: Sequence[tensor]) -> Sequence[tensor]
    >>> inputAxis = Axis('inputAxis')
    >>> @Function
    ... @Signature(input=SequenceOver[inputAxis][Tensor[128]], label=Tensor[10])
    ... def criterion(input, label):
    ...     output = model(input)
    ...     return cross_entropy_with_softmax(output, label)
    >>> print(criterion)
    Composite(input: SequenceOver[inputAxis][Tensor[128]], label: Tensor[10]) -> Tensor[1]

The following lists a few common errors with CNTK type objects:

Example:
    >>> # types are abstract, they cannot be instantiated directly
    >>> from cntk.layers.typing import Tensor
    >>> try:
    ...     inp = Tensor[32]()   # attempt to create an instance of type Tensor[32]
    ... except TypeError as e:
    ...     print('ERROR: ' + str(e))
    ERROR: abstract type Tensor[32] cannot be instantiated; use 'input_variable(**Tensor[32])' instead

    >>> # types are not inputs
    >>> try:
    ...     inp = Tensor[32]
    ...     y = sigmoid(inp)
    ... except ValueError as e:
    ...     print('ERROR: ' + str(e))
    ERROR: Input is a type object (Tensor[32]). Did you mean to pass 'input_variable(**Tensor[32])'?

    >>> # nested sequences are currently not supported
    >>> try:
    ...     t = Sequence[Sequence[Tensor[32]]]
    ... except TypeError as e:
    ...     print('ERROR: ' + str(e))
    ERROR: sequences over sequences are currently not supported

    >>> # a function with specified type gets passed a differently-shaped input
    >>> @Function
    ... @Signature(x=Tensor[13])
    ... def f(x):
    ...    return sigmoid(x)
    >>> try:
    ...     x = C.input_variable((42,))
    ...     y = f(x)
    ... except TypeError as e:
    ...     print('ERROR: ' + str(e))
    ERROR: argument x's type Tensor[13] is incompatible with the type Tensor[42] of the passed Variable

Using Python type syntax, besides being more concise and easier to memorize, has the added benefit of beign able to more easily talk about types of CNTK objects,
very similar to how one would talk about the types of Python objects (e.g. `List[Tuple[int,float]]`).
This is particularly beneficial for the functional-programming style of the Layers library, where functions are also reasoned about by their types.
In functional programming, it has been observed that getting the types of functions right is a critical step towards correct code.

Note that the type syntax does not allow to specify the special-purpose type property `needs_gradient`,
nor to `dtype` which instead should be specified as a global setting.
If these properties are needed on a type object, please use construct an input using :func:`~cntk.input_var` and get its `type` property.
'''

from ..axis import Axis
from ..variables import Variable, Record
from cntk.internal import sanitize_shape, _as_tuple
from cntk.internal.utils import get_python_function_arguments, map_function_arguments
import numpy as np

def _make_tensor_meta(cls_name, **kwargs):
    class TensorMeta(type):
        def __getitem__(self, shape):
            if not isinstance(shape, tuple):
                shape = (shape,)
            # the first shape parameter can be np.float32 or np.float64 or np.float16, similar to Eigen
            if len(shape) > 0 and (shape[0] == np.float32 or shape[0] == np.float64 or shape[0] == np.float16):
                kwargs['dtype'] = shape[0]
                shape = shape[1:]
            return Variable._Type(shape, **kwargs) # inject it for @Function 
    return TensorMeta(cls_name, (), {})

# Tensor and SparseTensor contain only a batch axis.
# If you want a sequence, say Sequence[tensor].
# ParameterTensor has no axis.
Tensor = _make_tensor_meta('Tensor', is_sparse=False, dynamic_axes=[Axis.default_batch_axis()])
'''
Meta class to denote a data tensor (with batch axis). Use with dimensions, e.g. ``Tensor[13,42]``.
'''
SparseTensor = _make_tensor_meta('SparseTensor', is_sparse=True, dynamic_axes=[Axis.default_batch_axis()])
'''
Meta class to denote a sparse data tensor (with batch axis). Use with dimensions, e.g. ``SparseTensor[129]``.
'''
ParameterTensor = _make_tensor_meta('ParameterTensor', is_sparse=False, dynamic_axes=[])
'''
Meta class to denote a parameter tensor (no batch axis). Use with dimensions, e.g. ``ParameterTensor[512,256]``.
'''

# Meta class to denote a data tensor (with batch axis) with unspecified dimensions.
tensor = Tensor[-2] # TODO: find the correct symbol for the sentinel value

def _make_seq_meta(cls_name, axes):
    class SeqMeta(type):
        def __getitem__(self, item_type):
            item_type = Variable._Type._sanitize(item_type)
            item_axes = getattr(item_type, 'dynamic_axes', None)
            if item_axes and item_axes != [Axis.default_batch_axis()]:
                raise TypeError('sequences over sequences are currently not supported')
            return Variable._Type(**item_type.updated_with(dynamic_axes=axes))
    return SeqMeta(cls_name, (), {})

Sequence = _make_seq_meta('Sequence', Axis.default_input_variable_dynamic_axes())
'''
Meta-meta class to denote a sequence of data tensors. Example: ``Sequence[Tensor[13,42]]``
'''
# TODO: accept Python's typing.Sequence instead; then import layers.typing by default in layers.__init__.py

class SequenceOverMeta(type):
    def __getitem__(self, axis):
        return _make_seq_meta('Sequence', [Axis.default_batch_axis(), axis])

SequenceOver = SequenceOverMeta('SequenceOver', (), {})
'''
Meta-meta-meta class to denote a sequence of data tensors over a custom axis. Example: ``userAxis = Axis(); SequenceOver[userAxis][Tensor[13,42]]``
'''


def Signature(*args, **kwargs):
    '''
    ``@Signature`` is a decorator to implement the function-argument annotations in Python-2.7,
    as needed by the ``@Function`` decorator.
    This is only needed when you have not yet migrated to Python 3.x.

    Note: Although this is aimed at enabling ``@Function`` syntax with type annotations
    in Python 2.7, ``@Signature`` is independent of CNTK and can be used for any argument annotation.

    Args:
        *args: types of arguments of the function that this decorator is applied to, in the same order.
        **kwargs: types of arguments with optional names, e.g. `x=Tensor[42]`. Use this second form for
           longer argument lists.

    Example::

     # Python 3:
     @Function
     def f(x: Tensor[42]):
         return sigmoid(x)

     # Python 2.7:
     @Function
     @Signature(Tensor[42])
     def f(x):
         return sigmoid(x)

     # note that this:
     @Function
     @Signature(x:int)
     def sqr(x):
         return x*x
     # is identical to:
     def sqr(x):
         return x*x
     sqr.__annotations__ = {'x': int}
    '''
    # this function returns another function which is the actual decorator applied to the def:
    def add_annotations(f):
        # prepare the signature
        param_names, annotations = get_python_function_arguments(f)
        if annotations:
            raise ValueError('@Signature cannot be applied to functions that already have annotations')
        annotations = {}
        if len(args) + len(kwargs) != len(param_names):
            raise TypeError("{} annotations provided for function to be decorated, but function has {} parameters".format(len(args) + len(kwargs), len(param_names)))
        # implant anotations into f
        params_dict = { name: name for name in param_names }
        f.__annotations__ = map_function_arguments(param_names, params_dict, *args, **kwargs)
        return f # and return the updated function
    return add_annotations
