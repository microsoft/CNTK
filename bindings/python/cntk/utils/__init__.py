# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numbers
import numpy as np
import scipy.sparse
from cntk import cntk_py

def precision_numpy(precision):
    '''
    Converts string precision to numpy precision 
    Args:
        precision (str): string precision

    Returns:
        numpy precision
    '''
    if precision == 'float':
        return np.float32
    elif precision == 'double':
        return np.float64
    else:
        raise ValueError('precision value: "%s" is not supported'%precision)

def cntk_device(device_id):
    '''
    Converts device ID to CNTK DeviceDescriptor instance
    Args:
        device_id (int): device id, -1 for CPU, 0 or higher for GPU

    Returns:
        CNTK DeviceDescriptor
    '''
    if device_id==-1:
        return cntk_py.DeviceDescriptor.cpu_device()
    else:
        return cntk_py.DeviceDescriptor.gpu_device(device_id)        

def cntk_to_numpy_shape(shape):
    '''
    Removes the dynamic axis and returns a tuple representing the NumPy shape.

    Args:
        shape (tuple or int): CNTK shape iterable

    Returns:
        a tuple that describes the NumPy shape of a tensor
    '''

    if np.isscalar(shape):
        shape = (shape,)    
    shape = shape[:-1]
    if not shape:
        shape = (1,)

    # cntk uses column major, thus we reverse the axes
    return tuple(reversed(shape))

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

def sanitize_shape(shape):
    """
    if shape is scalar create a tuple out of it and reverse it as cntk uses column major
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
    #if isinstance(arg, (Function, cntk_py.Function)):
    if isinstance(arg, cntk_py.Function):
        try:
            return arg.output()
        except RuntimeError:
            raise ValueError('the argument has more than one output, please provide the one you want')

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
    if len(args)==1 and isinstance(args, cntk_py.Function):
        args = [args]

    for arg in args:
        if isinstance(arg, (cntk_py.Variable, cntk_py.Constant, cntk_py.Parameter)):
            if cntk_py.DataType_Double == arg.get_data_type():
                dtypes.add(np.float64)
            elif cntk_py.DataType_Float == arg.get_data_type():
                dtypes.add(np.float32)
            else:
                raise ValueError('unknown data type')
        elif isinstance(arg, np.ndarray):
            if arg.dtype not in (np.float32, np.float64):
                raise ValueError('NumPy type "%s" is not supported'%arg.dtype)
                dtypes.add(arg.dtype)
        elif isinstance(arg, cntk_py.Function):
            var_outputs = arg.outputs()
            if len(var_outputs)>1:
                raise ValueError('expected single output, but got %i'%len(var_outputs))

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
    Z = np.zeros((len(batch), max_seq_len)+(data_point.shape), dtype=data_point.dtype)
    for idx, seq in enumerate(batch):
        if seq[0].shape != data_point.shape:
            raise ValueError('shape mismatch: expected %s but got '
                    ' %s'%(str(data_point.shape), str(seq[0].shape)))
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
        'array and not "%s"'%type(batch))

    seq_lens = [len(seq) for seq in batch]
    
    use_mask = len(set(seq_lens))!=1    
    if use_mask:
        # If not all sequences are of the same length, we have to pad them to
        # the same length and create a mask over the original data.
        from cntk.cntk_py import NDMask
        mask = NDMask((max(seq_lens), num_seq), device)
        for idx, seq_len in enumerate(seq_lens):
            mask.mask_section((seq_len, idx), (cntk_py.InferredDimension, 1)) 

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
            
    ndav = create_NDArrayView_from_NumPy(batch, device)

    if use_mask:
        value = Value(ndav, mask)
    else:
        value = Value(ndav)

    return value

def sanitize_var_map(input_map, precision_numpy=None, device=None, add_batch_axis=False):
    '''
    Sanitizes a dictionary of `Variable`s to input data such that it can be
    handed off to the `Forward` method.

    Args:
        input_map (`dict`): `Variable` to input (NumPy array or simple list of lists)
        precision_numpy : `np.float32`, `np.float64`, or `None`
        device (`DeviceDescriptor` or `None`): CNTK DeviceDescriptor
        add_batch_axis (`bool`): data in `input_map` are single instances and a batch axis has to be added

    Returns:
        `dict` that maps variables to sanitized batches
    '''
    var_map = {}
    if input_map:
        for var, batch in input_map.items():
            from ..cntk_py import Value
            if not isinstance(batch, Value):                
                if add_batch_axis:
                    batch = [batch]
                if isinstance(batch, np.ndarray):
                    if batch.dtype == np.int:
                        batch = batch.astype(np.float32)
                    if batch.dtype not in (np.float32, np.float64):                        
                        raise ValueError('only float32 and float64 are supported')
                    batch = sanitize_batch(batch, precision_numpy, device)
                else:
                    if is_tensor(batch):
                        batch = np.asarray(batch, dtype=precision_numpy)
                        batch = create_Value_from_NumPy(batch, device)
                    else:
                        batch = sanitize_batch(batch, precision_numpy, device)

            var_map[var] = batch

    return var_map

def remove_masked_elements(batch, mask):
    '''
    From a zero-padded `batch`, remove those entries that have a 0 in the
    `mask`. 

    Args:
        batch (`ndarray`): batch of samples that are variable length sequences padded by zeros to the max sequence length
        mask (`ndarray`): 2D matrix. Every row represents one sample. The columns have a `1` if the element is valid and `0` otherwise.

    Returns:
        a list of ndarrays
    '''
    return [seq[mask[idx]==1] for idx,seq in enumerate(batch)]

def ones_like(batch, precision_numpy):
    '''
    Returns a new batch, which has the same format as `batch` but all values
    set to 1.

    Args:
        batch (list of NumPy arrays): a list of sequences, which are NumPy arrays
    '''
    return [np.ones_like(sample, dtype=precision_numpy) for sample in batch]

def create_NDArrayView(shape, data_type=cntk_py.DataType_Float, dev=None):
    shape = sanitize_shape(shape)
    if not dev:
        dev = cntk_py.DeviceDescriptor.use_default_device()
    # FIXME only dense supported so far
    view = cntk_py.NDArrayView(data_type, cntk_py.StorageFormat_Dense, shape, dev)
    return view

def create_NDArrayView_from_NumPy(nd, dev=None):
    if not dev:
        dev = cntk_py.DeviceDescriptor.use_default_device()    
    view = cntk_py.NDArrayView(nd, dev, False)
    return view

def create_Value_for_Variable(var, shape=None, dev=None, mask=None):
    if not dev:
        dev = cntk_py.DeviceDescriptor.cpu_device()

    if shape is None:
        shape = var.shape().dimensions()
    view = cntk_py.NDArrayView(var.get_data_type(), cntk_py.StorageFormat_Dense, shape, dev)
    if mask:
        value = cntk_py.Value(view, mask)
    else:
        value = cntk_py.Value(view)
    return value

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

def sanitize_axis(rank, axis):
    if axis is None:
        return axis
    elif isinstance(axis, numbers.Integral):
        return cntk_py.Axis(rank - 1 - axis)
    elif axis.is_static_axis():
        return cntk_py.Axis(rank - 1 - axis.static_axis_index())
    else:
        return axis
     
def _py_dict_to_cntk_dict(py_dict):
    '''
    Converts a Python dictionary into a CNTK Dictionary whose values are CNTK DictionaryValue instances.
    Args:
        py_dict (dict): a dictionary to be converted.
    Returns: 
        :class:`cntk_py.Dictionary`
    '''
    res = cntk_py.Dictionary();
    for k,v in py_dict.items():
        if isinstance(v,dict):
            res[k] = cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(v))
        #TODO: add support to list of lists ?
        elif isinstance(v,list):
            l = list()
            for e in v:
                if isinstance(e,dict):
                    l.append(cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(e)))
                else:
                    l.append(cntk_py.DictionaryValue(v))
            res[k] = cntk_py.DictionaryValue(l)
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res
        
def create_minibatch_source(config_dict):
    '''
    Instantiate the CNTK built-in composite minibatch source which is used to stream data into the network.    
    Args:
        config_dict (dict): a dictionary containing all the key-value configuration entries.
    Returns: 
        :class:`cntk_py.MinibatchSource`
    '''
    cntk_dict = _py_dict_to_cntk_dict(config_dict)
    return cntk_py.create_composite_minibatch_source(cntk_dict)

def get_train_loss(trainer):
    '''
    Fetch the train loss from the last minibatch and copy it to the CPU in case it is on the GPU.
    Args:
        trainer (:class:`Trainer`): the trainer used.        
    Returns: 
        the loss value
    '''    
    import copy
    #we copy the value so swig does not destroy it when we leave the scope
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
    #we copy the value so swig does not destroy it when we leave the scope
    return copy.copy(trainer.previous_minibatch_evaluation_average())

def eval(op, precision, device, input_map=None, backward_pass=False):
    '''
    It evaluates `op` on the data provided by the reader. This is useful
    mainly to explore the operators and for convenient unit testing. 
    
    Args:
        op (:class:`Function`): operation to evaluate
        precision (`str` or `None`): precision being 'float32', 'float64', or `None`, in which case it will be determined by inspecting the operator (costly)
        device (:class:Cntk.DeviceDescriptor): the device the descriptor, whether it is CPU or GPU (and which one)
        input_map (`dict`): describes how to map inputs to the data in a data file using a number, NumPy array or reader object
        backward_pass (`bool`, optional): whether a backward pass is performed 

    Returns: 
        output generated by `op`. If `op` is an iterable, a dictionary
        op->result is returned. 
    '''
    if precision is not None:
        precision = precision_numpy(precision)

    forward_in_var_map = sanitize_var_map(input_map, precision, device)

    forward_out_var_map =  {}
    forward_retain = set()
    for v in op.outputs():
        forward_out_var_map[v] = None # will be populated in Forward()
        forward_retain.add(v)

    state = op.forward(forward_in_var_map, forward_out_var_map, device, forward_retain)

    forward_output = {}
    forward_output_mask = {}
    for v in op.outputs():
        value = forward_out_var_map[v]
        np_data = value.data().to_numpy()         
        if value.mask():
            np_data = remove_masked_elements(np_data, value.mask().to_numpy())
        forward_output[v] = np_data
        forward_output_mask[v] = value.mask()

    if backward_pass:    
        root_gradients = {} 
        for v, o in forward_output.items():
            root_gradients[v] = ones_like(o, precision)
        root_gradients = sanitize_var_map(root_gradients, precision, device)

        backward_var_map = dict((var, None) for var in forward_in_var_map)

        op.backward(state, root_gradients, backward_var_map)

        backward_output = {}
        for var, value in backward_var_map.items():
            np_data = value.data().to_numpy()             
            if value.mask():
                np_data = remove_masked_elements(np_data, value.mask().to_numpy())
            backward_output[var] = np_data

        return forward_output, backward_output

    else:
        return forward_output, None



