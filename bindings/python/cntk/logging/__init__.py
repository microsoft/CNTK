# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Utilities for logging. 
"""
from cntk import cntk_py
from .progress_print import *
from .graph import *
from enum import Enum, unique

@unique
class TraceLevel(Enum):
    '''
    Describes different logging verbosity levels.
    '''

    Error = cntk_py.TraceLevel_Error
    Warning = cntk_py.TraceLevel_Warning
    Info = cntk_py.TraceLevel_Info

    def __eq__(self, other):
        if isinstance(other, TraceLevel):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not (self == other)

def set_trace_level(value):
    '''
    Specifies global logging verbosity level.

    Args:
        value (:class:`~cntk.logging.TraceLevel`): required verbosity level.
    '''
    if isinstance(value, TraceLevel):
        cntk_py.set_trace_level(value.value)
    else:
        cntk_py.set_trace_level(value)

def get_trace_level():
    '''
    Returns current logging verbosity level.

    Returns:
        :class:`~cntk.logging.TraceLevel`: current verbosity level.
    '''
    return cntk_py.get_trace_level()