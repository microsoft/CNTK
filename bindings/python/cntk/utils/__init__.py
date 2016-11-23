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
from cntk.device import cpu, gpu, use_default_device
from .swig_helper import typemap
from ..axis import Axis
from .progress_print import *


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
    else:
        raise ValueError('precision value: "%s" is not supported' % precision)


def cntk_device(device_id):
    '''
    Converts the legacy device ID as it was used in CNTK 1 to a :class:`cntk.device.DeviceDescriptor` instance.

    Args:
        device_id (int): device id, -1 for CPU, 0 or higher for GPU

    Returns:
        :class:`cntk.device.DeviceDescriptor`
    '''
    if device_id == -1:
        return cpu()
    else:
        return gpu(device_id)


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
    Converts a list of NumPy arrays representing tensors of inputs into a
    format that is readable by :class:`~cntk.io.CTFDeserializer`.

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

@typemap
def one_hot(batch, num_classes, dtype=None, device=None): 
    '''
    Converts ``batch`` into a :class:`Value` object of ``dtype``
    such that the integer data in ``batch`` is interpreted as the indices
    representing one-hot vectors.

    Args:
        batch (list (of lists, if sequence) of index data): batch input data
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

    if dtype in [np.float32, None]: 
        value = cntk_py.Value.create_one_hot_float(num_classes, batch, device, False) 
    elif dtype == np.float64: 
        value = cntk_py.Value.create_one_hot_double(num_classes, batch, device, False) 
    return value

def has_seq_dim(var, data):
    '''
    Checks whether the data has a sequence dimensions or not. 

    By default, :func:`~cntk.ops.input_variable` sets up the input variable to
    have two dynamic axes, the batch and the sequence axis. When providing the
    data, the sequence axis can be left out, if the batch doesn't make use of
    sequences, and will be implicitly added during forward or backward pass. 

    Args:
        var (:class:`~cntk.ops.variables.Variable`): variable node for which
         the ``batch`` is meant
        data (list or NumPy array (dense, sparse)): batch input data

    Returns:
        whether ``data`` has a sequence axis
    '''
    num_dyn_axes = 0
    var_shape = var.shape
    var_rank = len(var_shape)

    # Find the innermost data sample
    drill = [data]
    drill_data = data
    if isinstance(drill_data, np.ndarray) or sparse.issparse(drill_data):
        drill_shape = drill_data.shape
    else:
        while isinstance(drill_data, list):
            drill_data = drill_data[0]
            num_dyn_axes += 1

            if isinstance(drill_data, np.ndarray) or sparse.issparse(drill_data):
                drill_shape = drill_data.shape
                break

            drill.append(drill_data)
            if isinstance(drill_data, Number):
                # Calculate the shape of the data point that would correspond to the input
                # variable's shape.
                drill_shape = _as_tuple(np.asarray(drill[-var_rank]).shape)
                drill.pop() 

                if drill_shape == ():
                    drill_shape = (1,)
                break

    if isinstance(drill_data, np.ndarray):
        # In case a full sequence is put inside an numpy array, we have
        # to account for the real sample shape.
        additional_dyn_axes = len(drill_shape) - var_rank
        num_dyn_axes += additional_dyn_axes
        drill_shape = drill_shape[additional_dyn_axes:]
    elif sparse.issparse(drill_data):
        if len(drill_shape)==2 and drill_shape[0]==1 and var_rank==1:
            # var_shape might be defined as e.g. (3,) or (1,3)
            drill_shape = (drill_shape[1],)
        elif len(drill_shape) > var_rank:
            # the sparse data might actually encode the full sequence so we
            sub_shape = drill_shape[-var_rank:]
            if sub_shape != var_shape:
                raise ValueError('data and variable shape do not match')
            num_dyn_axes += len(drill_shape) - var_rank
            drill_shape = var_shape

    # In drill_shape we now have potential data_points per drill level. We go
    # now backwards until the shape matches var_shape, at which point we have
    # found the real shape.
    while drill_shape!=var_shape and drill:
        drill_shape = np.asarray(drill.pop()).shape
        num_dyn_axes -= 1

    if drill_shape != var_shape:
        raise ValueError('could not match the data with shape %s to the '
                'input variable with shape %s'%(drill_shape, var_shape))

    num_var_dyn_axes = len(var.dynamic_axes)
    if num_dyn_axes == num_var_dyn_axes:
        return True
    elif num_dyn_axes == num_var_dyn_axes-1:
        return False
    else:
        raise ValueError(
        'data having %i axes is not compatible with the '
        'input variable having %i axes'%(num_dyn_axes,len(var_shape)))


def sanitize_shape(shape):
    """
    If shape is scalar, it creates a tuple out of it.
    """
    return _as_tuple(shape)


def sanitize_input(arg, fallback_dtype=np.float32, reshape=None):
    """
    Convert to :class:`cntk.ops.variables.Variable` so that it can be passed as Variable to the
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

    from cntk.ops.variables import Constant, Variable, Parameter
    from cntk.ops import constant

    # is it a Variable?
    if isinstance(arg,
                  (Constant, cntk_py.Constant,
                   Variable, cntk_py.Variable,
                   Parameter, cntk_py.Parameter)):
        return arg

    # or a Function?
    if isinstance(arg, cntk_py.Function):
        try:
            return arg.output
        except RuntimeError:
            raise ValueError(
                'the argument has more than one output, please provide the one you want')

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
        args (number, list, NumPy array, :class:`cntk.ops.variables.Variable`, 
         or :class:`cntk.ops.functions.Function`): input
    Returns:
        np.float32, np.float64, or None
    """
    from ..ops.variables import Variable

    cntk_dtypes = set()
    numpy_dtypes = set()
    if len(args) == 1 and isinstance(args, cntk_py.Function):
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
        elif isinstance(arg, cntk_py.Function):
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


def _pad_dense_to_max_len(var, batch, max_seq_len):
    """Appends the minimal required amount of zeroes at the end of each sample
    in the batch so that it becomes rectangular. ``batch`` is assumed to be
    row-major: first index is batch item, second is sequence item, then comes
    that actual sample. The sequence length is assumed to be the only varying
    dimension.

    Args:
        batch (list of NumPy arrays): list of arrays that differ only in their
         first dimension (different sequence lengths)
        max_seq_len (int): length to which the batch elements will be padded to

    Returns:
        Padded NumPy array
    """
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
            raise ValueError('shape mismatch: expected %s but got %s'
                             % (str(data_point.shape), str(seq[0].shape)))
        Z[idx, :len(seq)] += seq
    return Z

def _pad_sparse_seq_to_max_len(batch, max_seq_len):
    '''
    Appends sparse matrices of the same shape to every sequence so that they
    are of the same length.
    '''
    Z = []
    data_point = sparse.csr_matrix(batch[0][0].shape)
    for seq in batch:
        seq_len = seq.shape[0] if hasattr(seq, 'shape') else len(seq) 
        if seq_len>max_seq_len:
            raise ValueError('sequence of length %i exceeds max '
                    'length of %i'%(seq_len, max_seq_len))
        elif max_seq_len > seq_len:
            if isinstance(batch[0], list):
                seq += (max_seq_len-seq_len)*[data_point]
            else:
                shape = list(seq.shape)
                shape[0] = max_seq_len-seq_len
                seq = [seq, sparse.csr_matrix(tuple(shape))]
                
        if isinstance(seq, list):    
            seq = sparse.vstack(seq, format='csr')

        Z.append(seq)
    return Z

def _is_dense(batch):
    is_dense = True
    b = batch
    while isinstance(b, list):
        b = b[0]
        if sparse.issparse(b):
            return False

    return True

def sanitize_batch(var, batch, seq_starts=None, dtype=None, device=None):
    '''
    Convert to :class:`Value` with ``dtype``. If the samples in
    ``batch`` have different sequence lengths, pad them to max sequence length
    and create a mask.

    Args:
        var (:class:`~cntk.ops.variables.Variable`): variable node for which
         the ``batch`` is meant 
        batch (list of NumPy arrays): input
        seq_starts (list of bool or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the previous one (`False`)
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on

    Returns:
        :class:`Value`: converted batch that can be passed to the
         core API
    '''
    if isinstance(batch, cntk_py.Value):
        return batch

    # We need to figure out whether the data has a sequence axis. Note that
    # it is not enough to check whether the variable's dynamic axes include the
    # sequence axis, because the sequence axis might be omitted in the data if
    # it is not needed (CNTK core would then take care of this).
    batch_has_seq = has_seq_dim(var, batch)

    if isinstance(batch, list):
        is_dense = _is_dense(batch)

        if is_dense:
            seq_lens = [len(seq) for seq in batch]

            # If the input is a list of lists of dense values, all of the same
            # length, then we convert it into a NumPy array without requiring a
            # mask.
            if len(set(seq_lens)) == 1:
                batch = np.asarray(batch)
        else:
            if isinstance(batch[0], list):
                seq_lens = [len(seq) for seq in batch]
            else:
                seq_lens = [seq.shape[0] for seq in batch]

        if batch_has_seq:
            max_seq_len = max(seq_lens)
    else:
        is_dense = isinstance(batch, np.ndarray)
        # It is a sparse or dense NumPy array having all sequences being the
        # same length, so we just calculate the sequence lengths
        if batch_has_seq:
            max_seq_len = batch.shape[1]

    if dtype is None:
        dtype = get_data_type(var)

    if device is None:
        device = use_default_device()

    if isinstance(batch, np.ndarray):
        if np.issubdtype(batch.dtype, int):
            batch = batch.astype(var.dtype)
        elif batch.dtype not in (np.float32, np.float64):
            raise ValueError('only float32 and float64 are supported')

        ndav = create_NDArrayView_from_NumPy(batch, device)
        return Value(data=ndav)

    if isinstance(batch, list):
        if len(batch) == 0:
            raise ValueError('batch is empty')

        if not batch_has_seq and seq_starts is not None:
            raise ValueError('specification of individual sequence begins does not'
                    ' make sense when not using the sequence axis')

    # batch is now either a dense input that requires a mask, or it is sparse
    if batch_has_seq:
        mask = cntk_py.NDMask((len(batch), max_seq_len), 
                device or use_default_device())
        for idx, seq_len in enumerate(seq_lens):
            if seq_starts is None or seq_starts[idx]:
                mask.mark_sequence_begin((0, idx))
            # The first parameter is provided as a vector of ints, and thus
            # won't be automatically reversed to col-major, because of which we
            # provide it as such.
            # The second parameter is specifying the rectangle of the mask that
            # is invalid. As C++ is taking an NDShape, and we reverse the shape
            # in the SWIG layer, we provide it here as row-major.
            mask.invalidate_section((seq_len, idx),
                                    (1, cntk_py.InferredDimension))
    else:
        mask = None

    if is_dense:
        batch = _pad_dense_to_max_len(var, batch, max_seq_len)
        ndav = create_NDArrayView_from_NumPy(batch.astype(dtype), device)
        return Value(data=ndav, mask=mask)

    # There are three possibilities of providing sparse batches:
    # 1. batch is given as one big sparse array
    batch_is_sparse = sparse.issparse(batch) 
    if batch_is_sparse:
        sparse_tmp = batch
    else:
        # 2. batch is given as a list of sparse arrays, each of which is a full 
        #    sequence
        batch_has_sparse_sequences = batch_is_sparse or sparse.issparse(batch[0])
        if batch_has_sparse_sequences:
            sparse_tmp = batch[0]
        else:
            # 3. batch is given as a list of lists containing the sparse sequence
            #    elements
            batch_has_sparse_elements = batch_has_sparse_sequences or \
                    sparse.issparse(batch[0][0])
            if batch_has_sparse_elements:
                sparse_tmp = batch[0][0]

    if not sparse.isspmatrix_csr(sparse_tmp):
        raise ValueError("only CSR is supported as of now. Please " 
                "convert your data using 'batch.tocsr()'")

    if batch_is_sparse or batch_has_sparse_sequences or \
            batch_has_sparse_elements:

        batch_shape = batch.shape if hasattr(batch, 'shape') else (len(batch),)
        sample_shape = var.shape

        if not batch_is_sparse:
            # batch is not one big sparse matrix, but a list of them (or a list
            # of lists of them), so we have to create one. Two  possibilities:
            # 1. Batch has sequence axis: only 1d sparse vectors are allowed.
            # 2. Ohterwise, 1d or 2d sparse tensors are allowed
            if batch_has_seq:
                shape = batch[0][0].shape
                if not (len(shape)==1 or len(shape)==2 and shape[0]==1):
                    raise ValueError('only 1D sparse vectors are supported in ' 
                            ' sequence data, you gave shape %s'%str(shape))
                # Pad and stack the sparse vectors. 
                if batch_has_seq:
                    batch = _pad_sparse_seq_to_max_len(batch, max_seq_len)
                batch_shape += (max_seq_len,)
                # We are actually 1D. If rank==2, then the first dim is 1.
                sample_shape = sample_shape[-1]
            else:
                sample_shape = batch[0][0].shape
                if len(sample_shape) not in [1,2]:
                    raise ValueError('only 1D or 2D sparse vectors are supported')

            # Vertically stack sequences/samples
            batch = sparse.vstack(batch, format='csr')

            batch_shape += _as_tuple(sample_shape)

        ndav = cntk_py.NDArrayView(batch_shape, batch.data.astype(var.dtype),
                batch.indptr, batch.indices, device, False)

        return Value(data=ndav, mask=mask)

    else:
        raise ValueError('batch input not understood')


def sanitize_value(shape, value, dtype, device):
    '''
    Converts a given ``value`` to an :class:`NDArrayView` object that can be passed to
    the CNTK core.

    Args:
        shape (tuple): shape of the value
        value (None or value that can be cast to NumPy array): the value to
         be converted
        dtype: data type (np.float32 or np.float64)
        device (:class:`~cntk.device.DeviceDescriptor`): device this value should be put
         on

    Returns:
        :class:`~cntk.cntk_py.NDArrayView` object representing ``value``
    '''
    if value is None:
        if shape is None:
            raise ValueError('you need to specify at least shape or value')
        cntk_dtype = sanitize_dtype_cntk(dtype)
        ndav = create_NDArrayView(shape, cntk_dtype, device)
    else:
        np_dtype = sanitize_dtype_numpy(dtype)
        if not isinstance(value, np.ndarray) or value.dtype != np_dtype:
            if np.isscalar(value) and shape:
                value = np.full(shape, value, dtype=np_dtype)
            else:
                value = np.asarray(value, dtype=np_dtype)

        ndav = create_NDArrayView_from_NumPy(value, device)

    return ndav


def sanitize_function(arg):
    '''
    Tries to retrieve a Function from the argument or throws an exception if
    that's not possible.
    '''

    if isinstance(arg, cntk_py.Variable):
        arg = arg.owner

    if not isinstance(arg, cntk_py.Function):
        raise "Object of type %s cannot be cast to Variable" % str(type(arg))

    return arg


def sanitize_var_map(op_arguments, arguments, precision=None,
                     device=None):
    '''
    Sanitizes a dictionary of `Variable` s to input data such that it can be
    handed off to the evaluation methods (:meth:`cntk.ops.functions.Function.forward`, :meth:`cntk.ops.functions.Function.backward`, :meth:`cntk.Trainer.train_minibatch` and
    :meth:`cntk.Trainer.test_minibatch`).

    Args:
        op_arguments (:class:`cntk.ops.functions.Function`): arguments of the root function. In
         :meth:`cntk.ops.functions.Function.forward` pass it is typically `op.arguments`, in :meth:`cntk.ops.functions.Function.backward` pass it is
         `op.outputs`
        arguments: maps variables to their
         input data. The interpretation depends on the input type:
          * `dict`: keys are input variable or names and values are the input data.
          * any other type: if node has an unique input, ``arguments`` is mapped to this input.
            For nodes with more than one input, only `dict` is allowed.
         In both cases, every sample in the data will be interpreted
         as a new sequence. To mark samples as continuations of the
         previous sequence, specify ``arguments`` as `tuple`: the
         first element will be used as ``arguments``, and the second one will
         be used as a list of bools, denoting whether a sequence is a new
         one (`True`) or a continuation of the previous one (`False`).
         Data should be either NumPy arrays or a
         :class:`cntk.io.MinibatchData` instance.
        precision (str or `np.float32` or `np.float64`): if string it can be
         one of 'float' 'float32, 'double', 'float64', or None
        device (:class:`~cntk.device.DeviceDescriptor`, default None): device
         this value should be put on

    Returns:
        `dict` that maps variables to sanitized batches
    '''
    from ..io import MinibatchData

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

    sample_sizes = [v.shape[0] if hasattr(v, 'shape') else len(v) for v in arguments.values()]
    if len(set(sample_sizes)) != 1:
        raise ValueError('not all inputs have the same number of samples: ' +
                         ", ".join([str(s) for s in sample_sizes]))

    if seq_starts is not None:
        if not isinstance(seq_starts, (tuple, list)):
            raise ValueError(
                'if you specify seq_starts, it needs to be a list')

        sample_size = sample_sizes.pop()
        if len(seq_starts) != sample_size:
            raise ValueError('you have %i samples, but seq_starts has only %i' +
                             'elements' % (sample_sizes, len(seq_starts)))

    if precision is not None:
        precision = sanitize_precision(precision)

    var_map = {}
    for var, batch in arguments.items():
        if isinstance(var, str):
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

        if isinstance(batch, MinibatchData):
            batch = batch.m_data
        elif not isinstance(batch, cntk_py.Value):
            batch = sanitize_batch(var, batch, seq_starts, precision, device)

        var_map[var] = batch

    return var_map


def ones_like(batch, precision):
    '''
    Returns a new batch, which has the same format as ``batch`` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    return [np.ones_like(sample, dtype=sanitize_precision(precision)) for sample in batch]


def create_NDArrayView(shape, data_type=cntk_py.DataType_Float, device=None):
    shape = sanitize_shape(shape)
    if device is None:
        device = use_default_device()
    # FIXME only dense supported so far
    view = cntk_py.NDArrayView(data_type, cntk_py.StorageFormat_Dense, shape,
            device)
    return view


def create_NDArrayView_from_NumPy(nd, device=None):
    if device is None:
        device = use_default_device()

    return cntk_py.NDArrayView(nd, device, False)

class Value(cntk_py.Value):
    '''
    Internal representation of minibatch data.

    Args:
        shape (tuple): shape of the value
        value (None or value that can be cast to NumPy array): the value to
         be converted
        dtype: data type (np.float32 or np.float64)
        data (list or NumPy array (dense, sparse)): batch input data
        mask (list of list of bools): marks which elements are valid or not
         (used to mask out elements in ragged arrays)
        device (:class:`~cntk.device.DeviceDescriptor`): device this value should be put
         on
    '''
    def __init__(self, shape=None, dtype=None, data=None, mask=None, device=None):
        if device is None:
            device = use_default_device()

        if shape and dtype:
            ndav = create_NDArrayView(shape, dtype, device)

        elif data:
            if isinstance(data, np.ndarray):
                ndav = create_NDArrayView_from_NumPy(data, device)
            else:
                ndav = data

        if mask:
            super(Value, self).__init__(ndav, mask)
        else:
            super(Value, self).__init__(ndav)

    @property
    def shape(self):
        '''
        The rectangular shape of this value. I.e., if this value has sequences
        of varying lengths, the shape will have the max sequence length in the
        sequence dimension.
        '''
        return super(Value, self).shape().dimensions()

    def __len__(self):
        '''
        Number of samples in this value object.
        '''
        return self.shape[0]

def sanitize_dtype_numpy(dtype):
    is_type = isinstance(dtype, type) or isinstance(dtype, np.dtype)
    is_str = isinstance(dtype, str)
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
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_axis(axis):
    '''
    Sanitizes the axis.

    Args:
        axis (:class:`cntk.axis.Axis` or int or None): the axis to be used.

          * :class:`cntk.axis.Axis`: use axis instance directly (will convert row- to
             col-major in case of static axis.
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
    if axes != cntk_py.Axis.default_input_variable_dynamic_axes():
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

    np_data = np.asarray(value)
    mask = value.mask()
    if mask:
        mask = np.asarray(mask)
        np_data = [seq[mask[idx] != cntk_py.MaskKind_Invalid]
                   for idx, seq in enumerate(np_data)]

    return np_data


def eval(op, arguments=None, precision=None, device=None, backward_pass=False, expected_backward=None):
    '''
    It evaluates ``op`` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing.

    Args:
        op (:class:`Function`): operation to evaluate
        arguments: maps variables to their input data. The
         interpretation depends on the input type:
          * `dict`: keys are input variable or names, and values are the input data.
          * any other type: if node has an unique input, ``arguments`` is mapped to this input.
           For nodes with more than one input, only `dict` is allowed.
         In both cases, every sample in the data will be interpreted
         as a new sequence. To mark samples as continuations of the
         previous sequence, specify ``arguments`` as `tuple`: the
         first element will be used as ``arguments``, and the second one will
         be used as a list of bools, denoting whether a sequence is a new
         one (`True`) or a continuation of the previous one (`False`).
         Data should be either NumPy arrays or a
         :class:`cntk.io.MinibatchData` instance.
        seq_starts (list of `bool`s or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the previous one (`False`)
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

    state, forward_output = op.forward(arguments, op.outputs, op.outputs,
            device=device)

    if backward_pass:
        if expected_backward is None:
            expected_backward = arguments
        root_gradients = {v: ones_like(o, precision) for v, o in
                          forward_output.items()}

        backward_output = op.backward(state, root_gradients, expected_backward)

        return forward_output, backward_output

    else:
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
