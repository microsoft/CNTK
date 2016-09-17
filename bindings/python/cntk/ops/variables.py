import numpy as np
from cntk.cntk_py import NDArrayView, DeviceDescriptor, Variable, Parameter, ConstantFloat, ConstantDouble, Constant, DataType_Float, DataType_Double, ParameterFloat, ParameterDouble, Axis
from cntk import DATATYPE
from cntk.tensor import TensorOpsMixin
from .. import utils

FLOAT_32='float32'

def _sanitize_value(shape, value, dtype, device, is_param=False):
    np_dtype = utils.sanitize_dtype_numpy(dtype)
    cntk_dtype  = utils.sanitize_dtype_cntk(dtype)
    if value is None:
        if shape is None:
            raise ValueError('you need to specify at least shape or value')
        shape = utils.sanitize_shape(shape)

        if is_param:
            # TODO: expose the initialization params
            ndav = NDArrayView.random_uniform_float(shape, -0.05, 0.05, 1, device)
        else:
            ndav = utils.create_NDArrayView(shape, cntk_dtype, device)


    else:
        if not isinstance(value, np.ndarray) or value.dtype!=np_dtype:
            value = np.asarray(value, dtype=np_dtype)

        ndav = utils.create_NDArrayView_from_NumPy(value, device)

    return ndav

#TODO: remove default values from all constructors' arguments
class Variable(TensorOpsMixin,Variable):
    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False,
                    dynamic_axes = [Axis.default_dynamic_axis(), Axis.default_batch_axis()], name=''):
        shape = utils.sanitize_shape(shape)

        if data_type is None:
            data_type = FLOAT_32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse, dtype, needs_gradient, name, dynamic_axes)

class Parameter(TensorOpsMixin,Parameter):
    def __init__(self, shape=None, value=None, data_type=None, 
                    device=None, name=''):

        if data_type is None:
            if not isinstance(value, np.ndarray):
                data_type = FLOAT_32
            else:
                data_type = str(value.dtype)

        ndav = _sanitize_value(shape, value, data_type, device, True)
        super(Parameter, self).__init__(ndav, name)

# TODO: make this part of the above constructor
def parameter_from_scalar(value=None, data_type=None, 
                            device=None, name=''):
    if data_type is None:
        if not isinstance(value, np.ndarray):
            data_type = 'float32'
        else:
            data_type = str(value.dtype)

    dtype = utils.sanitize_dtype_cntk(data_type)

    if dtype == DataType_Float:
        return ParameterFloat((), value, device, name)
    elif dtype == DataType_Double:
        return ParameterDouble((), value, device, name)
    raise ValueError('unrecognized data_type: %s', dtype)

# TODO: make this part of the above constructor
class Constant(TensorOpsMixin,Constant):
    def __init__(self, shape=None, value=None, data_type=None, 
                    device=None, name=''):

        if data_type is None:
            if not isinstance(value, np.ndarray):
                data_type = FLOAT_32
            else:
                data_type = str(value.dtype)

        ndav = _sanitize_value(shape, value, data_type, device)
        super(Constant, self).__init__(ndav, name)

def constant_from_scalar(value=None, data_type=None,
                         device=None, name=''):
    if data_type is None:
        if not isinstance(value, np.ndarray):
            data_type = 'float32'
        else:
            data_type = str(value.dtype)

    dtype = utils.sanitize_dtype_cntk(data_type)
    
    if dtype == DataType_Float:
        return ConstantFloat((), value, device, name)
    elif dtype == DataType_Double:
        return ConstantDouble((), value, device, name)
    raise ValueError('unrecognized data_type: %s', dtype)
