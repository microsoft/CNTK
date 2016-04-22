import os
import sys
import numpy as np


def get_cntk_cmd():
    if "CNTK_EXECUTABLE_PATH" not in os.environ:
        raise ValueError(
            "you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

    return os.environ['CNTK_EXECUTABLE_PATH']


# Indent model description by how many spaces
MODEL_INDENTATION = 8

def cntk_to_numpy_shape(shape):
    '''
    Removes the sequence dimension.

    :param shape: CNTK shape iterable

    Returns a tuple that describes the NumPy shape of a tensor
    '''
    
    shape = tuple(int(s) for s in shape)

    shape = shape[:-1]
    if not shape:
        shape = (1,)

    return shape

def aggregate_readers(readers):
    import copy
    readers_map = {}
    for r in readers:
        filename = r['FileName']
        if filename in readers_map and\
                r.__class__.__name__ == readers_map[filename].__class__.__name__:
            readers_map[filename].inputs_def.extend(r.inputs_def)
        else:
            readers_map[filename] = copy.deepcopy(r)

    return [r for r in readers_map.values()]

def is_string(value):
    if sys.version_info.major<3:
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
    return ' '.join(data.ravel(order='F').astype(np.str))

def sparse_to_str(data):
    # return ' '.join('%s:%s'%(k,data[k]) for k in sorted(data.items()))
    raise NotImplementedError


def tensor_to_text_format(idx, alias, tensor, has_sequence_dimension=True):
    '''
    Converts a NumPy array representing tensor of one input into a format that
    is readable by `CNTKTextReader`.

    :param `alias`: alias to be used in the temporary file
    :param `tensor`: a NumPy array having sequence as its innermost dimension
    '''
    if not alias:
        raise ValueError('alias is missing')

    import scipy.sparse
    if isinstance(tensor, np.ndarray):
        to_str = dense_to_str
    elif scipy.sparse.issparse(tensor):
        raise ValueError('sparse is not yet supported')
        #to_str = sparse_to_str
    else:
        raise ValueError('sequence elements have to be of type numpy.ndarray' +
                ' (dense) or dictionary (sparse), you gave "%s"' % \
                str(type(tensor)))

    if has_sequence_dimension:
        num_seq_elements = tensor.shape[0]
        lines = []
        for seq_idx in range(0, num_seq_elements):
            lines.append('%i\t|%s %s'%(idx, alias, to_str(tensor[seq_idx])))

        return '\n'.join(lines)
    else:
        return '%i\t|%s %s'%(idx, alias, to_str(tensor))

def get_input_node(list_of_tensors, has_sequence_dimension, **kw):
    '''
    :param list_of_tensors: list of tensors potentially having sequences of
    different lengths.
    '''

    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?

    from cntk.context import get_context
    import tempfile

    # We have to use NamedTemporaryFile and close it, because the obvious first
    # choice, mkstemp(), would later fail in cntk.exe because the file would
    # still be locked.
    tf = tempfile.NamedTemporaryFile(prefix='_input_', suffix='.txt',
                                     dir=get_context().directory, delete=False)
    tf.close()

    if 'alias' in kw:        
        alias = kw['alias']
        del kw['alias']  # don't confuse with constructor's parameters
        
    if not alias:
        # TODO make sure we don't have clashes
        alias = '_I_%i' % np.random.randint(1000)

    shapes = set()
    with open(tf.name, 'w') as f:
        for idx,tensor in enumerate(list_of_tensors):
            if isinstance(tensor, list):
                tensor = np.asarray(tensor)

            if has_sequence_dimension:
                # collecting the shapes ignoring the sequence dimension
                shapes.add(tensor.shape[1:])
            else:
                shapes.add(tensor.shape)

            f.write(tensor_to_text_format(idx, alias, tensor,
                has_sequence_dimension) + '\n')

    # ignoring the sequence dimension, all shapes should be equal
    if len(shapes)!=1:
        raise ValueError('except for the sequence dimensions all shapes ' +
                'should be the same - instead we have: %s'%(", ".join(str(s) for s in shapes)))

    # shapes now contains only one shape, which has the sequence dimension
    # removed.
    value_shape = shapes.pop()

    cntk_shape = value_shape if value_shape else (1,)
    
    from ..ops import cntk1 
    node = cntk1.Input(cntk_shape, **kw)
    from ..reader import CNTKTextFormatReader
    node.reader = CNTKTextFormatReader(tf.name, alias)
        
    return node

def is_tensor(data):
    '''
    Checks whether the data is a tensor, i.e. whether it is a NumPy array or a
    list of NumPy arrays.

    :param `data`: data to check
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
            break

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
