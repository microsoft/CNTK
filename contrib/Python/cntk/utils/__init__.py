# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import numpy as np
import scipy.sparse

def get_cntk_cmd():
    if "CNTK_EXECUTABLE_PATH" not in os.environ:
        raise ValueError(
            "you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

    return os.environ['CNTK_EXECUTABLE_PATH']


# Indent model description by how many spaces
MODEL_INDENTATION = 4


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
        
def wrap_numpy_arrays(node):
    '''
    for a given computation node, wrapes its tensor inputs that are numpy arrays
    into input and constant nodes
    
    Args:
        node (:class:`cntk.graph.ComputationNode`): the computation node that will get its inputs wraped
    '''
    from ..graph import ComputationNode, _InputComputationNodeBase
    from ..ops import input_numpy, constant
    
    # The params are passed as arryas, e.g. plus([1,2], [3,4]),  and we need to 
    # wrap them with input and parameter nodes.
    first = True
    if node.params:
        for p in node.params:
            if p in node.inputs:
                val = getattr(node, p)
                #TODO: add support to tuple of numpy arrays, e.g. Splice(). 
                #So each tuple element will be wrapped
                if not (isinstance(val, ComputationNode) or isinstance(val, str)
                    or isinstance(val, tuple)):
                    # One param needs to be an Input() node. This will be fixed in 
                    # CNTK soon, so that we can remove this workaround and evaluate a 
                    # network with no inputs.
                    if first:        
                        ir = input_numpy([val])
                        setattr(node, p, ir)
                        first = False
                    else:
                        setattr(node, p, constant(getattr(node, p)))
                else:
                    if isinstance(val, _InputComputationNodeBase) and first:
                        first = False    