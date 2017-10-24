# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .swig_helper import typemap, map_if_possible
from .sanitize import *
from .sanitize import _as_tuple
import cntk

def _value_as_sequence(val, var):
    '''
    Helper function to hide map_if_possible().
    '''
    map_if_possible(val)
    return val.as_sequences(var)

def _value_as_sequence_or_array(val, var):
    has_seq_axis = len(var.dynamic_axes) > 1
    if has_seq_axis:
        return _value_as_sequence(val, var)
    else:
        map_if_possible(val)
        return val.asarray()

_serialization_version = 1

def _serialize(udf):
    dictionary = {}
    dictionary['class'] = udf.__class__.__name__
    dictionary['module'] = udf.__class__.__module__
    dictionary['op_name'] = udf.op_name
    dictionary['state'] = udf.serialize()
    dictionary['version'] = _serialization_version
    return dictionary


class _UDFDeserializeCallbackWrapper(cntk_py.UDFDeserializeCallbackWrapper):
    def __init__(self, factory_callback_map=None):
        super(_UDFDeserializeCallbackWrapper, self).__init__()
        self.factory_callback_map = factory_callback_map

    def __call__(self, inputs, name, dictionary):
        cls = dictionary['class']
        module = dictionary['module']
        state = dictionary['state']
        op_name = dictionary['op_name']
        deserialize_method = 'deserialize'

        if (self.factory_callback_map and op_name in self.factory_callback_map):
            factory = self.factory_callback_map[op_name]
        else:
            exec("from {} import {}".format(module, cls))
            eval_str = "{0}.{1} if hasattr({0}, '{1}') else None"
            factory = eval(eval_str.format(cls, deserialize_method))

        if factory:
            for i in range(len(inputs)):
                inputs[i].__class__ = cntk.Variable
            return factory(list(inputs), name, state)

        raise ValueError("Cannot deserialize user function '{}.{}'. "
            "It neither has a static 'deserialize' method, "
            "nor a factory callback was provided for the '{}' op name."
            .format(module, cls, op_name))

class _DeserializerFactory(cntk_py.DeserializerFactory):
    def __init__(self, callback):
        super(_DeserializerFactory, self).__init__()
        self.callback = callback

    def __call__(self, id):
        return self.callback(id)
