# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from .. import cntk_py

def map_if_possible(obj):
    from cntk.ops.variables import Variable, Parameter, Constant
    from cntk.ops.functions import Function
    from cntk.learner import Learner
    from cntk.io import MinibatchSource, MinibatchData, StreamConfiguration
    from cntk.axis import Axis
    typemap = { 
            cntk_py.Variable: Variable,
            cntk_py.Parameter: Parameter,
            cntk_py.Constant: Constant,
            cntk_py.Function: Function, 
            cntk_py.Learner: Learner, 
            cntk_py.MinibatchSource: MinibatchSource,
            cntk_py.MinibatchData: MinibatchData,
            cntk_py.StreamConfiguration: StreamConfiguration, 
            cntk_py.Axis: Axis,
            }
    # Some types like NumPy arrays don't let to set the __class__
    if obj.__class__ in typemap:
        obj.__class__ = typemap[obj.__class__]
            
def typemap(f):
    '''
    Upcasts Swig types to cntk types that inherit from Swig.
    '''
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwds):
        result = f(*args, **kwds)
        if isinstance(result, (tuple, list, set)):
            for r in result:
                map_if_possible(r)
        elif isinstance(result, dict):
            for k,v in result.items():
                map_if_possible(k)
                map_if_possible(v)
        else:
            try:
                map_if_possible(result)
            except TypeError:
                pass
        return result
    return wrapper
