# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numpy as np
import scipy.sparse


def cntk_to_numpy_shape(shape):
    '''
    Removes the dynamic axis and returns a tuple representing the NumPy shape.

    Args:
        shape (tuple): CNTK shape iterable

    Returns:
        a tuple that describes the NumPy shape of a tensor
    '''

    shape = tuple(int(s) for s in shape)

    shape = shape[:-1]
    if not shape:
        shape = (1,)

    # cntk uses column major, thus we reverse the axes
    return tuple(reversed(shape))


def aggregate_readers(readers):
    '''
    Aggregates the readers. If readers is provided, all elements have to
    reference the same filename. 

    Args:
        readers (iterable): readers to be aggregated
    '''
    import copy
    readers_map = {}

    reader_types = set([type(r) for r in readers])
    if len(reader_types) == 0:
        return None

    if len(reader_types) > 1:
        raise ValueError(
            'only one reader type is provided. You gave: %s' % str(reader_types))

    from ..reader import LazyInputReader, CNTKTextFormatReader
    if reader_types.pop() == LazyInputReader:
        from ..context import get_context
        filename = get_temp_filename(get_context().directory)
        r = CNTKTextFormatReader(filename)
        for lr in readers:
            r.add_lazy_input(lr)

        return r

    else:
        for r in readers:
            filename = r['FileName']
            if filename in readers_map and\
                    r.__class__.__name__ == readers_map[filename].__class__.__name__:
                readers_map[filename].inputs_def.extend(r.inputs_def)
            else:
                readers_map[filename] = copy.deepcopy(r)

        return list(readers_map.values())[0]


def is_string(value):
    if sys.version_info.major < 3:
        return isinstance(value, basestring)

    return isinstance(value, str)

# Copied from six


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})


def dense_to_str(data):
    return ' '.join(data.ravel(order='C').astype(np.str))


def sparse_to_str(data):
    return ' '.join('%s:%s'%(k,v) for k,v in sorted(data.items()))


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

def get_rank(shape):
    '''
    computes the rank of a tensor.
    
    Args:
        shape: it is either a tuple or an integer.
        
    Returns: the rank of the tensor.
    
    '''
    if np.isscalar(shape):
        if shape == 1:
            return 0
        else:
            return 1
    else:
        return len(shape)
        
import sys
from .. import cntk_py

def sanitize_input(arg):
    """
    Convert to Variable or Constant so that it can be passed as Variable to the CNTK
    operators. 
     * If `arg` is a NumPy array and its type is neither `np.float32` nor
    `np.float64`, it sets it to `np.float32`. 
     * If `arg` is an op, it is assumed that it has only one output, which will be returned.

    Args:
        arg (number, NumPy array, `Variable`, or `Function`): input

    Returns:
        Constant, if `arg` was a number or NumPy array. Variable otherwise.
    """

    from cntk.ops.variables import Constant, Variable, Placeholder
    if isinstance(arg, (Constant, Variable, Placeholder)):
        return arg

    try:
        var_output = arg.Output()
        if isinstance(var_output, Variable):
            return var_output
        else:
            raise ValueError('Cannot convert argument of type "%s" to Variable'%type(arg))
    except AttributeError:
        # no function or function with more then one output
        pass
    
    if isinstance(arg, list):
        if not arg:
            raise ValueError('input is empty')

        if not isinstance(arg[0], np.ndarray):
            raise ValueError('Cannot convert list of "%s" to Variable'%type(arg[0]))

    if not isinstance(arg, np.ndarray):
        arg = np.asarray(arg, dtype=np.float32)

    return Constant(value=arg)

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

    sample = np.asarray(batch[0][0])

    Z = np.zeros((len(batch), max_seq_len)+(sample.shape), dtype=sample.dtype)
    for idx, seq in enumerate(batch):
        Z[idx, :len(seq)] += seq 
    return Z

def sanitize_batch(batch, data_type, dev):
    """
    Convert to Value with mask.

    Args:
        batch (list of NumPy arrays): input

    Returns:
        Value
    """
    from ..cntk_py import Value

    if isinstance(batch, Value):
        return batch

    num_seq = len(batch)
    seq_lens = [len(seq) for seq in batch]

    # First we create the mask 
    from cntk.cntk_py import NDMask
    mask = NDMask((max(seq_lens), num_seq), dev)
    for idx, seq_len in enumerate(seq_lens):
        mask.MaskSection((seq_len, idx), (cntk_py.InferredDimension, 1)) 

    # Then we pad the batch to rectangular shape
    if isinstance(batch, list):
        if len(batch)==0:
            raise ValueError('batch is empty')

        batch = pad_to_dense(batch)

    # If it still is not an NumPy array, try brute force...
    if not isinstance(batch, np.ndarray):
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
            
    ndav_ptr = create_NDArrayViewPtr_from_NumPy(batch, dev)

    return Value(ndav_ptr, mask)


def create_NDArrayViewPtr(shape, data_type, dev):
    view = cntk_py.NDArrayView(data_type, cntk_py.StorageFormat_Dense, shape, dev)
    return view

def create_NDArrayViewPtr_from_NumPy(nd, dev):
    view = cntk_py.NDArrayView(nd, dev, False)
    return view

def create_ValuePtr_for_Variable(var, dev=None):
    if not dev:
        dev = cntk_py.DeviceDescriptor_CPUDevice()

    ndshape = var.Shape().Dimensions()+(1,1)
    view = cntk_py.NDArrayView(var.GetDataType(), cntk_py.StorageFormat_Dense, ndshape, dev)
    value = cntk_py.Value(view)
    return value

def create_ValuePtr(shape, data_type, dev):
    value = cntk_py.Value(create_NDArrayViewPtr(shape, data_type, dev))
    return value

def create_ValuePtr_from_NumPy(nd, dev):
    view_ptr = create_NDArrayViewPtr_from_NumPy(nd, dev)
    value = cntk_py.Value(view_ptr)
    return value

def sanitize_dtype_numpy(dtype):
    if dtype in ('float', 'float32', np.float32):
        return np.float32
    elif dtype in ('double', 'float64', np.float64):
        return np.float64
    else:
        raise ValueError('data type "%s" is not supported'%dtype)

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
        raise ValueError('data type "%s" is not supported'%dtype)

