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

