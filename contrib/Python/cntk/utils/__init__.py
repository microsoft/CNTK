import os
import sys

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
