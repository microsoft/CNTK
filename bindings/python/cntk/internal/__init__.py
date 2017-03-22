# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .swig_helper import typemap, map_if_possible
from .sanitize import *
from .sanitize import _as_tuple
from .. import cntk_py

_VARIABLE_OR_FUNCTION = (cntk_py.Variable, cntk_py.Function)

def get_data_type(*args):
    """
    Calculates the highest precision numpy data type of the provided parameters.
    If the parameter is a Function instance, it calculates it based on its
    inputs. Placeholders are ignored in the type determination.

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