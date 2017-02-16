# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
typing -- basic CNTK type meta-classes for CNTK @Function type signatures
'''

from ..axis import Axis
from ..ops.variables import Variable
from ..utils import Record, RecordWith

def _make_tensor_meta(cls_name, **kwargs):
    class TensorMeta(type):
        def __getitem__(self, shape):
            from ..utils import sanitize_shape
            shape = sanitize_shape(shape)
            return Variable.Type(shape, **kwargs) # inject it for @Function 
    return TensorMeta(cls_name, (), {})

# Tensor and SparseTensor contain only a batch axis.
# If you want a sequence, say Sequence[tensor].
# ParameterTensor has no axis.
Tensor          = _make_tensor_meta('Tensor',       is_sparse=False, dynamic_axes=Axis.default_batch_axis())
SparseTensor    = _make_tensor_meta('SparseTensor', is_sparse=True , dynamic_axes=Axis.default_batch_axis())
ParameterTensor = _make_tensor_meta('SparseTensor', is_sparse=True , dynamic_axes=[])
tensor = Tensor[-2] # TODO: find the correct symbol for the sentinel value

def _make_seq_meta(cls_name, axes):
    class SeqMeta(type):
        def __getitem__(self, item_type):
            return Variable.Type(**RecordWith(item_type, dynamic_axes=axes))
    return SeqMeta(cls_name, (), {})

Sequence = _make_seq_meta('Sequence', Axis.default_input_variable_dynamic_axes())
# TODO: accept Python's typing.Sequence instead; then import layers.typing by default in layers.__init__.py
# TODO: reject sequences over sequences (for now)

class SequenceOverMeta(type):
    def __getitem__(self, axis):
        return _make_seq_meta('Sequence', [Axis.default_batch_axis(), axis])

SequenceOver = SequenceOverMeta('SequenceOver', (), {})
