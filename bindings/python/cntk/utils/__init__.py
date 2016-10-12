# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numbers
import collections
import numpy as np
import scipy.sparse
from cntk import cntk_py
from .persist import *


def sanitize_precision(precision):
    '''
    Converts precision to NumPy precision

    Args:
        precision (`str` or `np.float32` or `np.float64`): precision, if string
         it can be one of 'float' 'float32, 'double', or 'float64'

    Returns:
        NumPy precision
    '''
    if precision in ['float', 'float32', np.float32]:
        return np.float32
    elif precision in ['double', 'float64', np.float64]:
        return np.float64
    else:
        raise ValueError('precision value: "%s" is not supported' % precision)


def cntk_device(device_id):
    '''
    Converts the legacy device ID as it was used in CNTK 1 to CNTK
    DeviceDescriptor instance.

    Args:
        device_id (int): device id, -1 for CPU, 0 or higher for GPU

    Returns:
        CNTK DeviceDescriptor
    '''
    if device_id == -1:
        return cntk_py.DeviceDescriptor.cpu_device()
    else:
        return cntk_py.DeviceDescriptor.gpu_device(device_id)


def is_string(value):
    if sys.version_info.major < 3:
        return isinstance(value, basestring)

    return isinstance(value, str)


def dense_to_str(data):
    return ' '.join(data.ravel(order='C').astype(np.str))


def sparse_to_str(data):
    return ' '.join('%s:%s' % (k, v) for k, v in sorted(data.items()))


def tensors_to_text_format(sample_idx, alias_tensor_map):
    '''
    Converts a list of NumPy arrays representing tensors of inputs into a format that
    is readable by `CNTKTextReader`.

    Args:
        sample_idx (int): number of current sample
        alias_tensor_map (dict): maps alias (str) to tensor (ndarray). Tensors
          are assumed to have dynamic axis.

    Returns:
        String representation in CNTKTextReader format
    '''

    max_seq_length = max(len(t) for t in alias_tensor_map.values())

    if max_seq_length == 0:
        return ''

    lines = []
    for seq_idx in range(0, max_seq_length):
        line = []

        for alias, tensor in sorted(alias_tensor_map.items()):
            if seq_idx >= len(tensor):
                # for this alias there no more sequence elements
                continue

            if is_tensor(tensor):
                if not isinstance(tensor, np.ndarray):
                    tensor = np.asarray(tensor)
                to_str = dense_to_str
            elif isinstance(tensor, list) and isinstance(tensor[0], dict):
                to_str = sparse_to_str
            else:
                raise ValueError(
                    'expected a tensor (dense) or list of dicts (sparse), but got "%s"' % type(tensor))

            line.append('%s %s' % (alias, to_str(tensor[seq_idx])))

        lines.append('%i\t|' % sample_idx + ' |'.join(line))

    return '\n'.join(lines)


def is_tensor(data):
    '''
    Checks whether the data is a tensor, i.e. whether it is a NumPy array or a
    list of NumPy arrays.

    Args:
        data: data to check

    Returns: True, if it is a tensor.
    '''
    if isinstance(data, np.ndarray):
        return True

    if not isinstance(data, list):
        return False

    while len(data) > 0:
        # All but the innermost dimension's values have to be lists
        try:
            data[0][0]
        except:
            # We reached the innermost dimension
            try:
                data[0] + 0
                return True
            except:
                # Innermost type is not a number
                return False

        if isinstance(data, np.ndarray):
            return True

        if not isinstance(data[0], list):
            return False

        data = data[0]

    return True


def is_tensor_list(data):
    '''
    Checks whether the data is a CNTK sequence, which is expressed in Python as
    a list of varying sized NumPy objects.
    '''
    is_list = isinstance(data, list)
    return is_list and len(data) > 0 and isinstance(data[0], np.ndarray)


def get_temp_filename(directory=None):
    '''
    Create and return a temporary filename.

    Args:
        directory (str): optional directory, in which the temporary file will
        be created

    Returns:
        Filename of the temporary file 
    '''
    import tempfile

    # We have to use NamedTemporaryFile and close it, because the obvious first
    # choice, mkstemp(), would later fail in cntk.exe because the file would
    # still be locked.
    tf = tempfile.NamedTemporaryFile(prefix='_input_', suffix='.txt',
                                     dir=directory, delete=False)
    tf.close()

    return tf.name


def sanitize_shape(shape):
    """
    If shape is scalar create a tuple out of it and reverse it as cntk uses column major
    """
    if np.isscalar(shape):
        shape = (shape,)
    return tuple(reversed(shape))


def sanitize_input(arg, fallback_dtype=np.float32):
    """
    Convert to Variable or Constant so that it can be passed as Variable to the CNTK
    operators. 
     * If `arg` is a NumPy array and its type is neither `np.float32` nor
    `np.float64`, it sets it to `np.float32`. 
     * If `arg` is an op, it is assumed that it has only one output, which will be returned.

    Args:
        arg (number, NumPy array, `Variable`, or `Function`): input
        fallback_dtype (numpy dtype): fallback dtype in case `arg` is a list

    Returns:
        Constant, if `arg` was a number or NumPy array. Variable otherwise.
    """

    from cntk.ops.variables import Constant, Variable
    from cntk.ops import constant

    # is it a Variable?
    if isinstance(arg, (Constant, Variable, cntk_py.Constant, cntk_py.Variable)):
        return arg

    # or a Function?
    # FIXME soon to be replaced by Function
    # if isinstance(arg, (Function, cntk_py.Function)):
    if isinstance(arg, cntk_py.Function):
        try:
            return arg.output()
        except RuntimeError:
            raise ValueError(
                'the argument has more than one output, please provide the one you want')

    # maybe a Python list that we can interpret as a NumPy array?
    if isinstance(arg, list) and not arg:
        raise ValueError('input is empty')

    if not isinstance(arg, np.ndarray):
        arg = np.asarray(arg, dtype=fallback_dtype)

    return constant(value=arg)


def get_data_type(*args):
    """
    Calculates the highest precision numpy datatype of the provided parameters. If
    the parameter is a Function instance, it calculates it based on its inputs.

    Args:
        args (number, `list`, NumPy array, `Variable`, or `Function`): input
    Returns:
        `np.float32` or `np.float64`
    """

    dtypes = set()
    if len(args) == 1 and isinstance(args, cntk_py.Function):
        args = [args]

    for arg in args:
        if isinstance(arg, (cntk_py.Variable, cntk_py.Value, cntk_py.NDArrayView)):
            if cntk_py.DataType_Double == arg.get_data_type():
                dtypes.add(np.float64)
            else:
                dtypes.add(np.float32)
        elif isinstance(arg, np.ndarray):
            if arg.dtype not in (np.float32, np.float64):
                raise ValueError(
                    'NumPy type "%s" is not supported' % arg.dtype)
                dtypes.add(arg.dtype)
        elif isinstance(arg, cntk_py.Function):
            var_outputs = arg.outputs()
            if len(var_outputs) > 1:
                raise ValueError(
                    'expected single output, but got %i' % len(var_outputs))

            var_output = var_outputs[0]
            if cntk_py.DataType_Double == var_output.get_data_type():
                dtypes.add(np.float64)
        else:
            # We don't know anything so we convert everything to float32. If it
            # works, we know the type.
            # TODO figure out a better/faster way.
            np.asarray(arg, dtype=np.float32)
            dtypes.add(np.float32)

    if np.float64 in dtypes:
        return np.float64
    else:
        return np.float32


def pad_to_dense(batch):
    """Appends the minimal required amount of zeroes at the end of each sample
    in the batch so that it becomes rectangular. `batch` is assumed to be
    row-major: first index is batch item, second is sequence item, then comes that
    actual sample. The sequence length is assumed to be the only varying
    dimension.

    Args:
        batch (list of NumPy arrays): list of arrays that differ only in their
        first dimension (different sequence lengths)

    Returns:
        Padded NumPy array
    """

    max_seq_len = max(len(r) for r in batch)

    # Assuming all sequences elements in all samples have the same shape
    data_point = np.asarray(batch[0][0])

    # FIXME
    # This is not the most efficient way of dealing with variable length
    # sequences, but so far the only one supported. Once, ragged arrays are
    # natively supported in CNTK, this will change.
    Z = np.zeros((len(batch), max_seq_len) +
                 (data_point.shape), dtype=data_point.dtype)
    for idx, seq in enumerate(batch):
        if seq[0].shape != data_point.shape:
            raise ValueError('shape mismatch: expected %s but got '
                             ' %s' % (str(data_point.shape), str(seq[0].shape)))
        Z[idx, :len(seq)] += seq
    return Z


def sanitize_batch(batch, data_type=None, device=None):
    """
    Convert to Value with `data_type`. If the samples in `batch` have different
    sequence lengths, pad them to max sequence length and create a mask.

    Args:
        batch (list of NumPy arrays): input

    Returns:
        converted batch
    """
    from ..cntk_py import Value

    if isinstance(batch, Value):
        return batch

    try:
        num_seq = len(batch)
    except TypeError:
        raise ValueError('expected an object of type Value or a NumPy ' +
                         'array and not "%s"' % type(batch))

    seq_lens = [len(seq) for seq in batch]

    use_mask = len(set(seq_lens)) != 1
    if use_mask:
        # If not all sequences are of the same length, we have to pad them to
        # the same length and create a mask over the original data.
        from cntk.cntk_py import NDMask
        mask = NDMask((max(seq_lens), num_seq), device)
        for idx, seq_len in enumerate(seq_lens):
            mask.mark_sequence_begin((0, idx))
            mask.invalidate_section((seq_len, idx), (cntk_py.InferredDimension, 1))

        # Then we pad the batch to rectangular shape
        if isinstance(batch, list):
            if len(batch) == 0:
                raise ValueError('batch is empty')

            batch = pad_to_dense(batch)

    # If it still is not an NumPy array, try brute force...
    if not isinstance(batch, np.ndarray):
        if data_type is None:
            data_type = np.float32
        batch = np.asarray(batch, dtype=data_type)

    '''
    if is_tensor(values) or is_tensor_list(values):
        values = np.asarray(values)
        if dynamic_axis:
            cntk_shape = values[0].shape[1:]
        else:
            cntk_shape = values[0].shape

        if len(cntk_shape) == 0:
            raise ValueError('values should be an array of input samples')
    '''

    ndav = create_NDArrayView_from_NumPy(batch, device)

    if use_mask:
        value = Value(ndav, mask)
    else:
        value = Value(ndav)

    return value

def sanitize_function(arg):
    '''
    Tries to retrieve a Function from the argument or throws an exception if
    that's not possible.
    '''
    if isinstance(arg, cntk_py.Variable):
        arg = arg.owner

    if not isinstance(arg, cntk_py.Function):
        raise "Object of type %s cannot be cast to Variable"%str(type(arg))

    return arg

def sanitize_var_map(op_arguments, arguments, precision=None, device=None, add_batch_axis=False):
    '''
    Sanitizes a dictionary of `Variable`s to input data such that it can be
    handed off to the `Forward` method.

    Args:
        op_arguments (`:class:Function`): arguments of the root function. In
         forward pass it is typically `op.arguments()`, in backward mode it is
         `op.outputs()`
        arguments (`dict` or `list`): 
          * map from input variables to the data
          * list of inputs in the order that the function expects or 
          Data should be either NumPy arrays or a `:class:cntk.io.MinibatchData` instance
        precision (`str` or `np.float32` or `np.float64`): if string it can be
         one of 'float' 'float32, 'double', 'float64', or `None` 
        device (`DeviceDescriptor` or `None`): CNTK DeviceDescriptor
        add_batch_axis (`bool`): data in `arguments` are single instances and a batch axis has to be added

    Returns:
        `dict` that maps variables to sanitized batches
    '''
    from ..cntk_py import Value
    from ..io import MinibatchData

    if arguments is None or isinstance(arguments, (list, tuple)) and len(arguments)==0:
        if len(op_arguments) > 0:
            raise ValueError('function expects %i arguments'%len(op_arguments))
        return {}

    if len(op_arguments) == 1 and not isinstance(arguments, dict):
        arguments = dict(zip(op_arguments, arguments))

    if isinstance(arguments, dict):
        arg_names = [var.name() for var in op_arguments]
        name_counter = collections.Counter(arg_names)

        var_name_map = dict((var.name(), var) for var in op_arguments)

    elif isinstance(arguments, list):
        arguments = dict(zip(op_arguments, arguments))

    else:
        raise ValueError('type "%s" is not supported'%type(arguments))

    if len(arguments) < len(op_arguments):
        raise ValueError('expected %i arguments, but got %i'%(len(op_arguments), len(arguments)))

    if precision is not None:
        precision = sanitize_precision(precision)

    var_map = {}
    for var, batch in arguments.items():
        if isinstance(var, str):
            if name_counter[var] == 0:
                raise ValueError('variable with name "%s" does not exist in the network. Available variable names: %s'%(var, ", ".join(var_name_map)))
            elif name_counter[var] > 1:
                raise ValueError('node name "%s" is not unique'%var)

            try:
                var = var_name_map[var]
            except KeyError:
                raise KeyError("no input with the name '%s' was found.  Available: %s"%(var, ", ".join(var_name_map.keys())))

        if isinstance(batch, MinibatchData):
            batch = batch.data()
        elif not isinstance(batch, Value):                
            if add_batch_axis:
                batch = [batch]
            if isinstance(batch, np.ndarray):
                if batch.dtype == np.int:
                    batch = batch.astype(np.float32)
                if batch.dtype not in (np.float32, np.float64):                        
                    raise ValueError('only float32 and float64 are supported')
                batch = sanitize_batch(batch, precision, device)
            else:
                if is_tensor(batch):
                    if precision is None:
                        precision = np.float32
                    batch = np.asarray(batch, dtype=precision)
                    batch = create_Value_from_NumPy(batch, device)
                else:
                    batch = sanitize_batch(batch, precision, device)

        var_map[var] = batch

    return var_map


def ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as `batch` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]


def create_NDArrayView(shape, data_type=cntk_py.DataType_Float, dev=None):
    shape = sanitize_shape(shape)
    if not dev:
        dev = cntk_py.DeviceDescriptor.use_default_device()
    # FIXME only dense supported so far
    view = cntk_py.NDArrayView(
        data_type, cntk_py.StorageFormat_Dense, shape, dev)
    return view


def create_NDArrayView_from_NumPy(nd, dev=None):
    if not dev:
        dev = cntk_py.DeviceDescriptor.use_default_device()

    return cntk_py.NDArrayView(nd, dev, False)


def create_Value(shape, data_type, dev):
    value = cntk_py.Value(create_NDArrayView(shape, data_type, dev))
    return value


def create_Value_from_NumPy(nd, dev):
    view = create_NDArrayView_from_NumPy(nd, dev)
    value = cntk_py.Value(view)
    return value


def sanitize_dtype_numpy(dtype):
    if dtype in ('float', 'float32', np.float32):
        return np.float32
    elif dtype in ('double', 'float64', np.float64):
        return np.float64
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_dtype_cntk(dtype):
    if dtype in (cntk_py.DataType_Float, cntk_py.DataType_Double,
                 cntk_py.DataType_Unknown):
        return dtype
    if dtype in ('float', 'float32', np.float32):
        return cntk_py.DataType_Float
    elif dtype in ('double', 'float64', np.float64):
        return cntk_py.DataType_Double
    elif not dtype:
        return cntk_py.DataType_Unknown
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_axis(rank, axis):
    if axis is None:
        return axis
    elif isinstance(axis, numbers.Integral):
        return cntk_py.Axis(rank - 1 - axis)
    elif axis.is_static_axis():
        return cntk_py.Axis(rank - 1 - axis.static_axis_index())
    else:
        return axis


def sanitize_dynamic_axes(axes):
    if axes is not cntk_py.Axis.default_input_variable_dynamic_axes:
        if not type(axes) in (list, tuple):
            axes = [axes]
        else:
            axes = tuple(reversed(axes))
    return axes


def get_train_loss(trainer):
    '''
    Fetch the train loss from the last minibatch and copy it to the CPU in case it is on the GPU.
    Args:
        trainer (:class:`Trainer`): the trainer used.
    Returns: 
        the loss value
    '''
    import copy
    # we copy the value so swig does not destroy it when we leave the scope
    return copy.copy(trainer.previous_minibatch_loss_average())


def get_train_eval_criterion(trainer):
    '''
    Fetch the train evaluation criterion (e.g., classification error) from the last minibatch and copy it to the CPU in case it is on the GPU.
    Args:
        trainer (:class:`Trainer`): the trainer used.
    Returns: 
        the criterion value
    '''
    import copy
    # we copy the value so swig does not destroy it when we leave the scope
    return copy.copy(trainer.previous_minibatch_evaluation_average())


def ensure_dev(ndav, dev):

    if ndav.device() != dev:

        ndav_on_target = create_NDArrayView(
            ndav.shape().dimensions(), data_type=ndav.get_data_type(), dev=dev)
        ndav_on_target.copy_from(ndav)
        ndav = ndav_on_target

    return ndav

def value_to_seq(value):
    '''
    Convert a Value to a sequence of NumPy arrays that have their masked
    entries removed.

    Args:
        value (`Value`): Value as it is returned by Swig

    Returns:
        a list of NumPy arrays
    '''

    np_data = value.data().to_numpy()         
    if value.mask():
        mask = value.mask().to_numpy()
        np_data = [seq[mask[idx] == 1] for idx, seq in enumerate(np_data)]

    return np_data

def eval(op, precision, device, arguments=None, backward_pass=False):
    '''
    It evaluates `op` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing. 

    Args:
        op (:class:`Function`): operation to evaluate
        precision (`str` or `None`): precision being 'float32', 'float64', or `None`, in which case it will be determined by inspecting the operator (costly)
        device (`:class:cntk.DeviceDescriptor`): the device the descriptor, whether it is CPU or GPU (and which one)
        arguments (`dict` or `list`): 
          * map from input variables to the data
          * list of inputs in the order that the function expects or 
          Data should be either NumPy arrays or a `:class:cntk.io.MinibatchData` instance
        backward_pass (`bool`, optional): whether a backward pass is performed 

    Returns: 
        output generated by `op`. If `op` is an iterable, a dictionary
        op->result is returned. 
    '''
    if precision is not None:
        precision = sanitize_precision(precision)

    forward_in_var_map = sanitize_var_map(op.arguments(), arguments, precision, device)

    forward_out_var_map = {}
    forward_retain = set()
    for v in op.outputs():
        forward_out_var_map[v] = None  # will be populated in Forward()
        forward_retain.add(v)

    state = op.forward(forward_in_var_map,
                       forward_out_var_map, device, forward_retain)

    forward_output = {}
    for v in op.outputs():
        forward_output[v] = value_to_seq(forward_out_var_map[v])

    if backward_pass:
        root_gradients = {}
        for v, o in forward_output.items():
            root_gradients[v] = ones_like(o, precision)
        root_gradients = sanitize_var_map(op.outputs(), root_gradients, precision, device)

        backward_var_map = dict((var, None) for var in forward_in_var_map)

        op.backward(state, root_gradients, backward_var_map)

        backward_output = {}
        for var, value in backward_var_map.items():
            backward_output[var] = value_to_seq(value)

        return forward_output, backward_output

    else:
        return forward_output, None


def typemap(f):
    '''
    Upcasts Swig types to cntk types that inherit from Swig.
    '''
            
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwds):
        from cntk.ops.variables import Variable, Parameter, Constant
        from cntk.ops.functions import Function
        from cntk.learner import Learner
        from cntk.io import MinibatchSource, MinibatchData, StreamConfiguration
        typemap = { 
                cntk_py.Variable: Variable,
                cntk_py.Parameter: Parameter,
                cntk_py.Constant: Constant,
                cntk_py.Function: Function, 
                cntk_py.Learner: Learner, 
                cntk_py.MinibatchSource: MinibatchSource,
                cntk_py.MinibatchData: MinibatchData,
                cntk_py.StreamConfiguration: StreamConfiguration, 
                }
        result = f(*args, **kwds)
        if isinstance(result, (tuple, list, set)):
            for r in result:
                r.__class__ = typemap.get(r.__class__, r.__class__)
        elif isinstance(result, dict):
            for k,v in result.items():
                k.__class__ = typemap.get(k.__class__, k.__class__)
                v.__class__ = typemap.get(v.__class__, v.__class__)
        else:
            result.__class__ = typemap.get(result.__class__, result.__class__)
        return result
    return wrapper
