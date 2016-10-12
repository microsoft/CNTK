import numpy as np
from cntk.cntk_py import NDArrayView, DeviceDescriptor, Variable, Parameter, ConstantFloat, ConstantDouble, Constant, DataType_Float, DataType_Double, ParameterFloat, ParameterDouble, Axis
from cntk import DATATYPE
from cntk.tensor import TensorOpsMixin
from .. import utils

FLOAT_32 = 'float32'


def _sanitize_value(shape, value, dtype, device):
    np_dtype = utils.sanitize_dtype_numpy(dtype)
    cntk_dtype = utils.sanitize_dtype_cntk(dtype)

    if value is None:
        if shape is None:
            raise ValueError('you need to specify at least shape or value')
        shape = utils.sanitize_shape(shape)
        ndav = utils.create_NDArrayView(shape, cntk_dtype, device)
    else:
        if not isinstance(value, np.ndarray) or value.dtype != np_dtype:
            if np.isscalar(value) and shape:
                value = np.full(shape, value, dtype=np_dtype)
            else:
                value = np.asarray(value, dtype=np_dtype)

        ndav = utils.create_NDArrayView_from_NumPy(value, device)

    return ndav


class Variable(TensorOpsMixin, Variable):

    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False,
                 dynamic_axes=[Axis.default_dynamic_axis(), Axis.default_batch_axis()], name=''):
        shape = utils.sanitize_shape(shape)

        if data_type is None:
            data_type = FLOAT_32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse,
                                       dtype, needs_gradient, name, dynamic_axes)


class Parameter(TensorOpsMixin, Parameter):

    def __init__(self, shape=None, init=None, data_type=None,
            device=None, name=''):

        if data_type is None:
            if not isinstance(init, np.ndarray):
                data_type = FLOAT_32
            else:
                data_type = str(init.dtype)

        if init is None:
            init = 0

        if isinstance(init, (np.ndarray, list, float, int)):
            ndav = _sanitize_value(shape, init, data_type, device)
            super(Parameter, self).__init__(ndav, name)

        else:
            shape = utils.sanitize_shape(shape)
            data_type  = utils.sanitize_dtype_cntk(data_type)
            super(Parameter, self).__init__(shape, data_type, init,
                    device, name)

class Constant(TensorOpsMixin, Constant):

    def __init__(self, shape=None, value=None, data_type=None, 
                    device=None, name=''):

        if data_type is None:
            if not isinstance(value, np.ndarray):
                data_type = FLOAT_32
            else:
                data_type = str(value.dtype)

        ndav = _sanitize_value(shape, value, data_type, device)
        super(Constant, self).__init__(ndav, name)
