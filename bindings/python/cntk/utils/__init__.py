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
from cntk.device import use_default_device, cpu
from .swig_helper import typemap
from ..axis import Axis
from .progress_print import *

_VARIABLE_OR_FUNCTION = (cntk_py.Variable, cntk_py.Function)

def is_string(s):
    '''
    Tests whether ``s`` is a string in a way that works on Python 2 and 3.
    '''
    return isinstance(s, ("".__class__, u"".__class__))

def sanitize_precision(precision):
    '''
    Converts precision to NumPy precision

    Args:
        precision (str or `np.float32` or `np.float64`): precision, if string
         it can be one of 'float' 'float32, 'double', or 'float64'

    Returns:
        NumPy precision
    '''
    if precision in [cntk_py.DataType_Float, 'float', 'float32', np.float32]:
        return np.float32
    elif precision in [cntk_py.DataType_Double, 'double', 'float64', np.float64]:
        return np.float64
    elif precision in [cntk_py.DataType_Unknown]:
        return None
    else:
        raise ValueError('precision value: "%s" is not supported' % precision)


@typemap
def one_hot(batch, num_classes, dtype=None, device=None):
    '''
    Converts ``batch`` into a :class:`~cntk.core.Value` object of ``dtype``
    such that the integer data in ``batch`` is interpreted as the indices
    representing one-hot vectors.

    Example:
        >>> num_classes = 6
        >>> sparse_indices = [[1,5],[4]]
        >>> i0 = C.input_variable(shape=num_classes, is_sparse=True)
        >>> z = C.times(i0, np.eye(num_classes))
        >>> value = C.one_hot(sparse_indices, num_classes)
        >>> z.eval({i0: value})
        [array([[ 0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  1.]], dtype=float32), array([[ 0.,  0.,  0.,  0.,  1.,  0.]], dtype=float32)]

    Args:
        batch (list of lists of integers): batch input data of indices
        num_classes (int): number of classes
        dtype (`np.float32`, `np.float64`, default None): data type
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on

    Returns:
        ``batch`` converted into a :class:`~Value` object that can be passed to
        the forward or eval function.
    '''
    if device is None:
        device = use_default_device()

    if isinstance(batch, np.ndarray):
        batch = batch.tolist()

    try:
        data_type = type(batch[0][0])
    except:
        raise ValueError('input must be a list of list of integers')

    if data_type != int:
        raise ValueError('supplied data to one_hot() must be of type integer'
                ' and not "%s" since it is index data.'%data_type)

    if dtype in [np.float32, None]:
        value = cntk_py.Value.create_one_hot_float(num_classes, batch, device, False)
    elif dtype == np.float64:
        value = cntk_py.Value.create_one_hot_double(num_classes, batch, device, False)
    return value


def sanitize_shape(shape):
    """
    If shape is scalar, it creates a tuple out of it.
    """
    return _as_tuple(shape)


def sanitize_input(arg, fallback_dtype=np.float32, reshape=None):
    """
    Convert to :class:`~cntk.ops.variables.Variable` so that it can be passed as Variable to the
    CNTK operators.

      * If ``arg`` is a NumPy array and its type is neither `np.float32` nor `np.float64`, it sets it to `np.float32`.
      * If ``arg`` is an op, it is assumed that it has only one output, which will be returned.

    Args:
        arg (number, NumPy array, :class:`~cntk.ops.variables.Variable`, or :class:`~cntk.ops.functions.Function`): input
        fallback_dtype (NumPy dtype): fallback dtype in case ``arg`` is a list

    Returns:
      Leaves Constant, Parameter, and Variable as is. Returns Constant, if
      ``arg`` is a number or NumPy array. Variable otherwise.
    """

    from cntk.ops.functions import UserFunction
    from cntk.ops.variables import Constant, Variable, Parameter
    from cntk.ops.functions import Function
    from cntk.ops import constant

    # is it a Variable or a Function?
    if isinstance(arg,
                  (Constant, cntk_py.Constant,
                   Variable, cntk_py.Variable,
                   Parameter, cntk_py.Parameter,
                   Function, cntk_py.Function)):
        return arg

    # maybe a Python list that we can interpret as a NumPy array?
    if isinstance(arg, list) and not arg:
        raise ValueError('input is empty')

    if not isinstance(arg, np.ndarray) or arg.dtype!=fallback_dtype:
        arg = np.asarray(arg, dtype=fallback_dtype)
        if arg.shape == ():
            arg.shape = (1,)

    if reshape:
        arg = np.reshape(arg, reshape)

    return constant(value=arg)


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
        if isinstance(arg, Variable) and arg.is_placeholder==True:
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

@typemap
def sanitize_batch(var, batch, seq_starts=None, device=None):
    '''
    Convert to :class:`~cntk.core.Value`.

    Args:
        var (:class:`~cntk.ops.variables.Variable`): input variable into which
         ``batch`` is passed
        batch: batch input for `var`. It can be
         * a single NumPy array denoting the full minibatch
         * a list of NumPy arrays or SciPy sparse CSR matrices each representing a sequence
         * a :class:`~cntk.core.Value` object (e.g. returned by :func:`one_hot`)
        seq_starts (list of `bool`s or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans one for each sequence in the batch that tell whether a
         sequence is a new sequence (`True`) or a continuation of the sequence
         in the same slot of the previous minibatch (`False`)
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on

    Returns:
        :class:`~cntk.core.Value`: converted batch that can be passed to the core API
    '''
    if isinstance(batch, cntk_py.Value):
        if seq_starts is not None:
            raise ValueError('for directly passed Value objects sequence '
                    'starts cannot be used yet.')
        return batch

    if seq_starts and len(var.dynamic_axes)<=1:
        raise ValueError('you specified sequence begin markers, but your '
                'input_variable does not contain a sequence axis.')

    if device is None:
        device = use_default_device()

    from .. import Value
    return Value.create(var, batch, seq_starts, device)


def sanitize_value(shape, value, dtype, device):
    '''
    Converts a given ``value`` to an :class:`~cntk.NDArrayView` object that can be passed to
    the CNTK core.

    Args:
        shape (tuple): shape of the value
        value (None or value that can be cast to NumPy array): the value to
         be converted
        dtype: data type (np.float32 or np.float64)
        device (:class:`~cntk.device.DeviceDescriptor`): device this value should be put
         on

    Returns:
        :class:`~cntk.NDArrayView` object representing ``value``
    '''
    from .. import NDArrayView
    if value is None:
        if shape is None:
            raise ValueError('you need to specify at least shape or value')
        cntk_dtype = sanitize_dtype_cntk(dtype)
        ndav = NDArrayView(shape, cntk_dtype, device)
    else:
        np_dtype = sanitize_dtype_numpy(dtype)
        if not isinstance(value, np.ndarray) or value.dtype != np_dtype:
            if np.isscalar(value) and shape:
                value = np.full(shape, value, dtype=np_dtype)
            else:
                value = np.asarray(value, dtype=np_dtype)

        ndav = NDArrayView.from_dense(value, device)

    return ndav


def sanitize_function(arg):
    '''
    Tries to retrieve a Function from the argument or throws an exception if
    that's not possible.
    '''
    from cntk.ops import combine

    if isinstance(arg, cntk_py.Variable):
        arg = combine([arg])

    if not isinstance(arg, cntk_py.Function):
        raise TypeError("Object of type %s cannot be cast to Variable" %
                str(type(arg)))

    return arg

def sanitize_var_map(op_arguments, arguments, precision=None,
                     device=None, extract_values_from_minibatch_data=True):
    '''
    Sanitizes a dictionary of `Variable` s to input data such that it can be
    handed off to the evaluation methods
    (:meth:`~cntk.ops.functions.Function.forward`,
    :meth:`~cntk.ops.functions.Function.backward`, :meth:`~cntk.Trainer.train_minibatch` and
    :meth:`~cntk.Trainer.test_minibatch`).

    Args:
        op_arguments (:class:`~cntk.ops.functions.Function`): arguments of the root function. In
         :meth:`~cntk.ops.functions.Function.forward` pass it is typically
         `op.arguments`, in :meth:`~cntk.ops.functions.Function.backward` pass it is
         `op.outputs`
        arguments: maps variables to their input data. The interpretation depends on
         the input type:

           * dict: keys are input variable or names, and values are the input data.
           * any other type: if node has an unique input, arguments is
             mapped to this input.
         For nodes with more than one input, only dict is allowed.

         In both cases, every every sample in the data will be interpreted
         as a new sequence.

         Sequences can be marked as continuations of the same sequence in
         the previous minibatch (that is the sequence in the same slot).
         There are two possibilities for this:

          * specifying arguments as a `tuple` where the first element is
            used as arguments and the second one will be used as a list
            of bools, denoting whether a sequence is a new one (`True`) or a
            continuation of the sequence in the same slot of the previous
            minibatch (`False`). This will be applied to all batches.
          * specifying arguments as a dictionary of variables to tuples
            where the first element is used as arguments and the second
            one will be used as a list of bools, denoting whether a sequence
            is a new one (`True`) or a continuation of the sequence in the
            same slot of the previous minibatch (`False`). This will be
            applied to all batches.

         Data should be either NumPy arrays or a
         :class:`~cntk.io.MinibatchData` instance.
        precision (str or `np.float32` or `np.float64`): if string it can be
         one of 'float' 'float32, 'double', 'float64', or None
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on
        extract_values_from_minibatch_data (`bool`, defaults to `True`): specifies
         if :class:`~cntk.io.MinibatchData` instances in the arguments map are
         converted to the underlying value (:class:`~cntk.core.Value`) instances (default),
         or if they should remain intact, as they contain additional meta
         information required by the Trainer (specifically, by the
         :meth:`~cntk.Trainer.train_minibatch` method).

    Returns:
        `dict` that maps variables to sanitized batches
    '''
    from ..io import MinibatchData

    if not op_arguments:
        return {}

    if isinstance(arguments, tuple):
        arguments, seq_starts = arguments
    else:
        seq_starts = None

    if arguments is None or isinstance(arguments, (dict, list)) and len(arguments) == 0:
        if len(op_arguments) > 0:
            raise ValueError('function expects %i arguments' %
                             len(op_arguments))
        return {}

    if len(arguments) < len(op_arguments):
        raise ValueError('your graph has %i inputs, but you specified %i' %
                        (len(op_arguments), len(arguments)))

    if isinstance(arguments, dict):
        arg_names = [var.name for var in op_arguments]
        name_counter = collections.Counter(arg_names)

        var_name_map = dict((var.name, var) for var in op_arguments)
    else:
        if len(op_arguments) == 1:
            name_counter = collections.Counter([op_arguments[0].name])
            var_name_map = dict([(op_arguments[0].name, op_arguments[0])])
            arguments = dict([(op_arguments[0], arguments)])
        else:
            raise ValueError('non-dict argument (%s) is not supported for nodes with more than one input' % type(arguments).__name__)

    if precision is not None:
        precision = sanitize_precision(precision)

    var_map = {}
    for var, batch in arguments.items():
        if is_string(var):
            if name_counter[var] == 0:
                raise ValueError('variable with name "%s" does not exist in the network. Available variable names: %s' % (
                    var, ", ".join(var_name_map)))
            elif name_counter[var] > 1:
                raise ValueError('node name "%s" is not unique' % var)

            try:
                var = var_name_map[var]
            except KeyError:
                raise KeyError("no input with the name '%s' was found.  Available: %s" % (
                    var, ", ".join(var_name_map.keys())))

        if isinstance(batch, tuple):
            if seq_starts is not None:
                raise ValueError('you cannot provide sequence start '
                        'information globally and for individual batches '
                        'at the same time')

            batch, seq_starts = batch

            if seq_starts is not None:
                if not isinstance(seq_starts, (tuple, list)):
                    raise ValueError(
                        'if you specify sequence begin markers, it needs to be a list')

                sample_size = batch.shape[0] if hasattr(batch, 'shape') else len(batch)

                if len(seq_starts) != sample_size:
                    raise ValueError('you have %i sequences, but only %i '
                            'sequence begin markers' % (sample_size, len(seq_starts)))

        if seq_starts is not None and isinstance(batch, cntk_py.Value):
            raise ValueError('for directly passed Value objects sequence '
                    'starts cannot be used yet.')

        if isinstance(batch, MinibatchData) and extract_values_from_minibatch_data:
            batch = batch.data

        if not (isinstance(batch, MinibatchData) or isinstance(batch, cntk_py.Value)):
            batch = sanitize_batch(var, batch, seq_starts, device)

        var_map[var] = batch

    return var_map


def _ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as ``batch`` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]

def sanitize_dtype_numpy(dtype):
    is_type = isinstance(dtype, type) or isinstance(dtype, np.dtype)
    is_str = is_string(dtype)
    if is_type and dtype in (int, np.float32) or \
            hasattr(dtype, 'kind') and dtype.kind in 'iu' \
            or is_str and dtype in ('float', 'float32'):
        return np.float32
    elif is_type and dtype in (float, np.float64) or \
            is_str and dtype in ('double', 'float64'):
        # The Python type 'float' is a np.float64
        return np.float64
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_dtype_cntk(dtype):
    if isinstance(dtype, int) and dtype in (cntk_py.DataType_Float, cntk_py.DataType_Double, cntk_py.DataType_Unknown):
        return dtype
    if dtype is None:
        return cntk_py.DataType_Unknown

    dtype = sanitize_dtype_numpy(dtype)
    if dtype == np.float32:
        return cntk_py.DataType_Float
    elif dtype == np.float64:
        return cntk_py.DataType_Double
    elif dtype == object:
        return cntk_py.DataType_Unknown
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_axis(axis):
    '''
    Sanitizes the axis.

    Args:
        axis (:class:`~cntk.axis.Axis` or int or None): the axis to be used.

          * :class:`~cntk.axis.Axis`: use axis instance directly (will convert
            row- to col-major in case of static axis).
          * int: if positive, use it as static axis. If negative, count from
            last to first axis
          * None: denote all available axes
    '''
    if axis is None:
        return Axis.all_static_axes()
    elif isinstance(axis, numbers.Integral):
        return Axis(-axis - 1)
    elif axis.is_static_axis:
        return Axis(-1 - axis.static_axis_index())
    else:
        return axis


def sanitize_dynamic_axes(axes):
    if not type(axes) in (list, tuple):
        axes = [axes]
    for ax in axes:
        if not isinstance(ax, cntk_py.Axis):
            raise TypeError('type Axis expected, got %s instead'%type(ax))
    axes = tuple(reversed(axes))
    return axes


def get_train_loss(trainer):
    '''
    Fetch the train loss from the last minibatch and copy it to the CPU in case it is on the GPU.

    Args:
        trainer (:class:`~cntk.trainer.Trainer`): the trainer used.
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


# Obsolete: All usages should be replaced with the variable_value_to_seq procedure below
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
        value (:class:`~cntk.core.Value`): Value as it is returned by Swig

    Returns:
        a list of NumPy arrays
    '''

    mask = value.mask()
    if mask:
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
        state, forward_output = op.forward(arguments, op.outputs, None, device=device)
        return forward_output, None

# helper to convert a dictionary into a Python class, so that the dict looks like an immutable record
# TODO: move to utils?
class _ClassFromDict(dict):
    def __init__(self, args_dict):
        super(_ClassFromDict, self).__init__(args_dict)
        # TODO: try to delete __setattr__ to make it immutable
        self.__dict__.update(args_dict)
        #for key in args_dict:   # self.__dict__.update(args_dict)
        #    self[key] = args_dict[key]
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("record has no attribute '{}'".format(key))
        return self[key]
    def __setattr__(self, key, value):
        raise AttributeError('record is immutable')


# easier construction of records
# e.g. r = Record(x = 13, y = 42) ; x = r.x
def Record(**kwargs):
    return _ClassFromDict(kwargs)

def _as_tuple(x):
    '''
    Convert an argument to a tuple.

    Args:
        x: if scalar, it returns ``(x,)``. If iterable, it converts it to
        tuple.

    Returns:
        Tuple of ``x``.
    '''
    if np.isscalar(x):
        x = (x,)
    return tuple(x)

def start_profiler(dir='profiler', sync_gpu=True, reserve_mem=cntk_py.default_profiler_buffer_size):
    '''
    Start profiler to prepare performance statistics gathering. Note that profiler is not enabled after start.
	[Example](https://github.com/Microsoft/CNTK/wiki/Performance-Profiler#for-python)

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
