# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
typing -- basic CNTK type meta-classes for CNTK @Function type signatures
'''

from ..axis import Axis
from ..variables import Variable, Record
from cntk.internal import sanitize_shape
from cntk.internal.utils import get_python_function_arguments, map_function_arguments

def _make_tensor_meta(cls_name, **kwargs):
    class TensorMeta(type):
        def __getitem__(self, shape):
            shape = sanitize_shape(shape)
            return Variable.Type(shape, **kwargs) # inject it for @Function 
    return TensorMeta(cls_name, (), {})

# Tensor and SparseTensor contain only a batch axis.
# If you want a sequence, say Sequence[tensor].
# ParameterTensor has no axis.
# BUGBUG: Scalars cannot be described since Tensor[] is invalid. Use 'float'?
Tensor          = _make_tensor_meta('Tensor',       is_sparse=False, dynamic_axes=Axis.default_batch_axis())
'''
Meta class to denote a data tensor (with batch axis). Use with dimensions, e.g. ``Tensor[13,42]``.
'''
SparseTensor    = _make_tensor_meta('SparseTensor', is_sparse=True , dynamic_axes=Axis.default_batch_axis())
'''
Meta class to denote a sparse data tensor (with batch axis). Use with dimensions, e.g. ``SparseTensor[129]``.
'''
ParameterTensor = _make_tensor_meta('ParameterTensor', is_sparse=False , dynamic_axes=[])
'''
Meta class to denote a parameter tensor (no batch axis). Use with dimensions, e.g. ``ParameterTensor[512,256]``.
'''
tensor = Tensor[-2] # TODO: find the correct symbol for the sentinel value
'''
Meta class to denote a data tensor (with batch axis) with unspecified dimensions.
'''

def _make_seq_meta(cls_name, axes):
    class SeqMeta(type):
        def __getitem__(self, item_type):
            return Variable.Type(**item_type.updated_with(dynamic_axes=axes))
    return SeqMeta(cls_name, (), {})

Sequence = _make_seq_meta('Sequence', Axis.default_input_variable_dynamic_axes())
'''
Meta-meta class to denote a sequence of data tensors. Example: ``Sequence[Tensor[13,42]]``
'''
# TODO: accept Python's typing.Sequence instead; then import layers.typing by default in layers.__init__.py
# TODO: reject sequences over sequences (for now)

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
     sqr.__annotations__ = {'x': int}``
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