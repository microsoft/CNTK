# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py
import numpy as np

_VARIABLE_OR_FUNCTION = (cntk_py.Variable, cntk_py.Function)

def get_data_type(*args):
    """
    Calculates the highest precision numpy data type of the provided parameters.
    If the parameter is a Function instance, it calculates it based on its
    inputs. placeholders are ignored in the type determination.

    Args:
        args (number, list, NumPy array, :class:`~cntk.variables.Variable`, or :class:`~cntk.ops.functions.Function`): input

    Returns:
        np.float32, np.float64, or None
    """
    from ..variables import Variable

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
            from ..variables import Record

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

def _ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as ``batch`` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    from cntk.internal import sanitize_precision
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]


def eval(op, arguments=None, precision=None, device=None, backward_pass=False, expected_backward=None):
    '''
    It evaluates ``op`` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing.

    Args:
        op (:class:`~cntk.ops.functions.Function`): operation to evaluate
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

def _py_dict_to_cntk_dict(py_dict):
    '''
    Recursively converts a Python dictionary into a CNTK Dictionary 
    whose values are CNTK DictionaryValue instances.

    Args:
        py_dict (dict): a dictionary to be converted.

    Returns:
        cntk_py.Dictionary:
        A :class:`~cntk.cntk_py.Dictionary` that has been converted from the input `dict`
    '''
    from ..cntk_py import DictionaryValueFromDict, DictionaryValue, Dictionary
    def _to_cntk_dict_value(py_value):
        if isinstance(py_value, dict):
            return DictionaryValueFromDict(_py_dict_to_cntk_dict(py_value))
    
        if isinstance(py_value, list):
            py_list = list(map(_to_cntk_dict_value, py_value))
            return DictionaryValue(py_list)
    
        return DictionaryValue(py_value)

    res = Dictionary()
    for k, v in py_dict.items():
        res[k] = _to_cntk_dict_value(v)
    return res