# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from functools import wraps
from .. import cntk_py

_typemap = None
def map_if_possible(obj):
    global _typemap
    if _typemap is None:
        # We can do this only if cntk_py and the cntk classes are already
        # known, which is the case, when map_if_possible is called.
        from cntk.ops.variables import Variable, Parameter, Constant
        from cntk.ops.functions import Function
        from cntk.learners import Learner
        from cntk.train.trainer import Trainer
        from cntk.train.training_session import TrainingSession
        from cntk.io import MinibatchSource, MinibatchData, StreamConfiguration
        from cntk.axis import Axis
        from cntk.train.distributed import WorkerDescriptor, Communicator, DistributedLearner
        from cntk import Value, NDArrayView
        _typemap = {
                cntk_py.Variable: Variable,
                cntk_py.Parameter: Parameter,
                cntk_py.Constant: Constant,
                cntk_py.Function: Function,
                cntk_py.Learner: Learner,
                cntk_py.Value: Value,
                cntk_py.NDArrayView: NDArrayView,
                cntk_py.MinibatchSource: MinibatchSource,
                cntk_py.Trainer: Trainer,
                cntk_py.TrainingSession: TrainingSession,
                cntk_py.MinibatchData: MinibatchData,
                cntk_py.StreamConfiguration: StreamConfiguration,
                cntk_py.Axis: Axis,
                cntk_py.DistributedWorkerDescriptor: WorkerDescriptor,
                cntk_py.DistributedCommunicator: Communicator,
                cntk_py.DistributedLearner: DistributedLearner
                }

    # Some types like NumPy arrays don't let to set the __class__
    if obj.__class__ in _typemap:
        obj.__class__ = _typemap[obj.__class__]
    else:
        if isinstance(obj, (tuple, list, set)):
            for o in obj:
                map_if_possible(o)
        elif isinstance(obj, dict):
            for k,v in obj.items():
                map_if_possible(k)
                map_if_possible(v)

def typemap(f):
    '''
    Decorator that upcasts return types from Swig types to cntk types that
    inherit from Swig. It does so recursively, e.g. if the return type is a
    tuple containing a dictionary, it will try to upcast every element in the
    tuple and all the keys and values in the dictionary.
    '''
    @wraps(f)
    def wrapper(*args, **kwds):
        result = f(*args, **kwds)
        map_if_possible(result)
        return result
    return wrapper
