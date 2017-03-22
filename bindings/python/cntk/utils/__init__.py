# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import numbers
import collections
import copy
import numpy as np
from numbers import Number
from scipy import sparse

from .. import cntk_py
from ..device import use_default_device, cpu
from ..axis import Axis
from cntk.internal import typemap
from cntk import core

# To __remove__
from cntk.logging import *
# End to remove

_VARIABLE_OR_FUNCTION = (cntk_py.Variable, cntk_py.Function)


# To __remove__
def one_hot(batch, num_classes, dtype=None, device=None):
    import cntk
    return cntk.Value.one_hot(batch, num_classes, dtype, device)
# End to remove


def get_data_type(*args):
    """
    Calculates the highest precision numpy data type of the provided parameters.
    If the parameter is a Function instance, it calculates it based on its
    inputs. Placeholders are ignored in the type determination.

    Args:
        args (number, list, NumPy array, :class:`~cntk.ops.variables.Variable`, or :class:`~cntk.ops.functions.Function`): input

    Returns:
        np.float32, np.float64, or None
    """
    from ..ops.variables import Variable

    cntk_dtypes = set()
    numpy_dtypes = set()
    if len(args) == 1 and isinstance(args, _VARIABLE_OR_FUNCTION):
        args = [args]

    for arg in args:
        if isinstance(arg, Variable) and arg.is_placeholder == True:
            continue
        if isinstance(arg,
                      (cntk_py.Variable, cntk_py.Value, cntk_py.NDArrayView)):
            if cntk_py.DataType_Double == arg.get_data_type():
                cntk_dtypes.add(np.float64)
            elif cntk_py.DataType_Float == arg.get_data_type():
                cntk_dtypes.add(np.float32)
        elif isinstance(arg, np.ndarray):
            if arg.dtype not in (np.float32, np.float64):
                raise ValueError(
                    'NumPy type "%s" is not supported' % arg.dtype)
            numpy_dtypes.add(arg.dtype.type)
        elif isinstance(arg, _VARIABLE_OR_FUNCTION):
            var_outputs = arg.outputs
            if len(var_outputs) > 1:
                raise ValueError(
                    'expected single output, but got %i' % len(var_outputs))

            var_type = var_outputs[0].get_data_type()
            if cntk_py.DataType_Double == var_type:
                cntk_dtypes.add(np.float64)
            else:
                cntk_dtypes.add(np.float32)
        else:
            # We don't know anything so we convert everything to float32. If it
            # works, we know the type.
            # TODO figure out a better/faster way.
            np.asarray(arg, dtype=np.float32)
            numpy_dtypes.add(np.float32)

    if cntk_dtypes:
        if np.float64 in cntk_dtypes:
            return np.float64
        elif np.float32 in cntk_dtypes:
            return np.float32
    else:
        if np.float64 in numpy_dtypes:
            return np.float64
        elif np.float32 in numpy_dtypes:
            return np.float32


def _is_dense(batch):
    if isinstance(batch, np.ndarray):
        return True
    elif sparse.issparse(batch):
        return False

    is_dense = True
    b = batch
    while isinstance(b, list):
        b = b[0]
        if sparse.issparse(b):
            return False

    return True




def _ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as ``batch`` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    from cntk.internal import sanitize_precision
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]




def get_train_loss(trainer):
    '''
    Fetch the train loss from the last minibatch and copy it to the CPU in case it is on the GPU.

    Args:
        trainer (:class:`~cntk.train.trainer.Trainer`): the trainer used.
    Returns:
        the loss value
    '''
    # we copy the value so swig does not destroy it when we leave the scope
    return copy.copy(trainer.previous_minibatch_loss_average)


def get_train_eval_criterion(trainer):
    '''
    Fetch the train evaluation criterion (e.g., classification error) from the last minibatch and copy it to the CPU in case it is on the GPU.

    Args:
        trainer (:class:`Trainer`): the trainer used.
    Returns:
        the criterion value
    '''
    # we copy the value so swig does not destroy it when we leave the scope
    return copy.copy(trainer.previous_minibatch_evaluation_average)


# Obsolete: All usages should be replaced with the variable_value_to_seq
# procedure below
def value_to_seq(value):
    '''
    Convert a Value to a sequence of NumPy arrays that have their masked
    entries removed.

    Args:
        value (:class:`~cntk.core.Value`): Value as it is returned by Swig

    Returns:
        a list of NumPy arrays
    '''

    np_data = np.asarray(value)
    mask = value.mask()
    if mask:
        mask = np.asarray(mask)
        np_data = [seq[mask[idx] != cntk_py.MaskKind_Invalid]
                   for idx, seq in enumerate(np_data)]

    return np_data


def variable_value_to_seq(value, variable):
    '''
    Convert a Value to a sequence of NumPy arrays that have their masked
    entries removed.

    Args:
        value (:class:`~cntk.cntk_py.Value` or :class:`~cntk.core.Value`): Value object

    Returns:
        a list of NumPy arrays
    '''

    if isinstance(value, core.Value):
        has_mask = value.mask.any()
    elif isinstance(value, cntk_py.Value):
        has_mask = value.mask() is not None

    if has_mask:
        value_sequences = value.unpack_variable_value(variable, True, cpu())
        return [np.asarray(seq) for seq in value_sequences[0]]
    else:
        return np.asarray(value)


def eval(op, arguments=None, precision=None, device=None, backward_pass=False, expected_backward=None):
    '''
    It evaluates ``op`` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing.

    Args:
        op (:class:`Function`): operation to evaluate
        arguments: maps variables to their input data. The
         interpretation depends on the input type:

          * `dict`: keys are input variable or names, and values are the input data.

          * any other type: if node has a unique input, ``arguments`` is mapped to this input.
            For nodes with more than one input, only `dict` is allowed.

         In both cases, every sample in the data will be interpreted
         as a new sequence. To mark samples as continuations of the
         previous sequence, specify ``arguments`` as `tuple`: the
         first element will be used as ``arguments``, and the second one will
         be used as a list of bools, denoting whether a sequence is a new
         one (`True`) or a continuation of the previous one (`False`).
         Data should be either NumPy arrays or a
         :class:`~cntk.io.MinibatchData` instance.
        seq_starts (list of bools or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the sequence in the same slot of the previous
         minibatch (`False`)
        precision (str or None): precision being 'float32', 'float64', or
         None, in which case it will be determined by inspecting the operator
         (costly)
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on
        backward_pass (`bool`, optional): whether a backward pass is performed
        expected_backward (`dict` or None): keys are variables for which to
         compute a backward ouptut. By default (None) all entries from
         'arguments' are used

    Returns:
        mapping of output variables to their values.
    '''

    if backward_pass:
        state, forward_output = op.forward(arguments, op.outputs, op.outputs,
                                           device=device)

        if expected_backward is None:
            expected_backward = arguments
        root_gradients = {v: _ones_like(o, precision) for v, o in
                          forward_output.items()}

        backward_output = op.backward(state, root_gradients, expected_backward)

        return forward_output, backward_output

    else:
        state, forward_output = op.forward(
            arguments, op.outputs, None, device=device)
        return forward_output, None

class Record(dict):
    '''
    Easy construction of a record (=immutable singleton class) from keyword arguments.
    e.g. r = Record(x = 13, y = 42) ; x = r.x

    Args:
        kwargs: keyword arguments to turn into the record members

    Returns:
        A singleton class instance that has all passed kw args as immutable class members.
    '''
    def __init__(self, **args_dict):
        super(Record, self).__init__(args_dict)
        self.__dict__.update(args_dict)
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("record has no attribute '{}'".format(key))
        return self[key]

    def __setattr__(self, key, value):
        raise AttributeError('record is immutable')
    def updated_with(self, **kwargs):
        '''
        Create a new Record from an existing one with members modified or added.
        e.g. r = Record(x = 13) ; print(r.x) ; r2 = r.updated_with(x = 42) ; print(r2.x)
    
        Args:
            kwargs: keyword arguments to turn into the record members
    
        Returns:
            A singleton class instance that has all passed kw args as immutable class members.
        '''
        d = dict(**self)   # make it mutable
        d.update(kwargs)   # merge the new items
        return Record(**d) # lock it up again

def get_python_function_arguments(f):
    '''
    Helper to get the parameter names and annotations of a Python function.
    '''
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    import sys
    if sys.version_info.major >= 3:
        from inspect import getfullargspec
    else:
        def getfullargspec(f):
            from inspect import getargspec
            annotations = getattr(f, '__annotations__', {})
            #f.__annotations__ = None  # needed when faking it under Python 3 for debugging purposes
            a = getargspec(f)
            #f.__annotations__ = annotations
            return Record(args=a.args, varargs=a.varargs, varkw=a.keywords, defaults=a.defaults, kwonlyargs=[], kwonlydefaults=None, annotations=annotations)
    param_specs = getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    defaults = param_specs.defaults # "if this tuple has n elements, they correspond to the last n elements listed in args"
    if defaults:
        arg_names = arg_names[:-len(defaults)] # we allow Function(functions with default arguments), but those args will always have default values since CNTK Functions do not support this
    return (arg_names, annotations)

def map_function_arguments(params, params_dict, *args, **kwargs):
    '''
    Helper to determine the argument map for use with various call operations.
    Returns a dictionary from parameters to whatever arguments are passed.
    Accepted are both positional and keyword arguments.
    This mimics Python's argument interpretation, except that keyword arguments are not optional.
    This does not require the arguments to be Variables or Functions. It is also called by train_minibatch() and @Signature.
    '''
    # start with positional arguments
    arg_map = dict(zip(params, args))

    # now look up keyword arguments
    if len(kwargs) != 0:
        for name, arg in kwargs.items():  # keyword args are matched by name
            if name not in params_dict:
                raise TypeError("got an unexpected keyword argument '%s'" % name)
            param = params_dict[name]
            if param in arg_map:
                raise SyntaxError("got multiple values for argument '%s'" % name)
            arg_map[param] = arg # add kw argument to dict
    assert len(arg_map) == len(params)

    return arg_map

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




def start_profiler(dir='profiler', sync_gpu=True, reserve_mem=cntk_py.default_profiler_buffer_size):
    '''
    Start profiler to prepare performance statistics gathering. Note that
    the profiler is not enabled after start
    (`example
    <https://github.com/Microsoft/CNTK/wiki/Performance-Profiler#for-python>`_).

    Args:
        dir: directory for profiler output
        sync_gpu: whether profiler syncs CPU with GPU when timing
        reserve_mem: size in byte for profiler memory reserved
    '''
    cntk_py.start_profiler(dir, sync_gpu, reserve_mem)


def stop_profiler():
    '''
    Stop profiler from gathering performance statistics and flush them to file
    '''
    cntk_py.stop_profiler()


def enable_profiler():
    '''
    Enable profiler to gather data. Note that in training_session, profiler would be enabled automatically after the first check point
    '''
    cntk_py.enable_profiler()


def disable_profiler():
    '''
    Disable profiler from gathering data.
    '''
    cntk_py.disable_profiler()


