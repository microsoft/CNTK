# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .swig_helper import typemap, map_if_possible
from .sanitize import *
from .sanitize import _as_tuple

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

class UserFunctionDeserializer(cntk_py.UDFDeserializer):
    '''
    Provides an implementation of the UDFDeserializer interface used
    to inflate user defined functions in a model dictionary.
    '''
    def __init__(self, factory_callback_map=None):
        super(UserFunctionDeserializer, self).__init__()
        self.factory_callback_map = factory_callback_map
        self.__disown__()

    def _deserialize(self, inputs, name, dictionary):
        cls = dictionary['class']
        module = dictionary['module']
        state = dictionary['state']
        op_name = dictionary['op_name']

        if (self.factory_callback_map and op_name in self.factory_callback_map):
            factory = self.factory_callback_map[op_name]
        else:
            exec("from {} import {}".format(module, cls))
            eval_str = "{0}.deserialize if hasattr({0}, 'deserialize') else None"
            factory = eval(eval_str.format(cls))

        if (factory):
            return factory(list(inputs), name, state)

        raise ValueError("Cannot deserialize user function '{}.{}'. "
            "It neither has a static 'deserialize' method, "
            "nor a factory callback was provided for the '{}' op name."
            .format(module, cls, op_name))