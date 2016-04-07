import os

if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError(
        "you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']

# Indent model description by how many spaces
MODEL_INDENTATION = 8


def numpy_to_cntk_shape(shape):
    '''
    Converting the NumPy shape (row major) to CNTK shape (column major).

    :param shape: NumPy shape tuple

    Returns a tuple that can be ':'.join()ed to a CNTK dimension.
    '''
    if not shape:
        # in case of a scalar
        return (1,)

    return tuple(reversed(shape))

def cntk_to_numpy_shape(shape):
    '''
    Converts col-major to row-major and removes the sequence dimension.

    :param shape: CNTK shape iterable

    Returns a tuple that describes the NumPy shape of a tensor
    '''
    
    shape = tuple(int(s) for s in reversed(shape))

    shape = shape[1:]
    if not shape:
        shape = (1,)

    return shape

def dedupe_readers(readers):
    import copy
    readers_map = {}
    for r in readers:
        filename = r['FileName']
        if filename in readers_map:
            readers_map[filename].inputs_def.extend(r.inputs_def)
        else:
            readers_map[filename] = copy.deepcopy(r)

    return [r for r in readers_map.values()]

