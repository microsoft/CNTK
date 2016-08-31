import numpy as np
from cntk import DATATYPE, NDArrayView, DeviceDescriptor_cpudevice, DeviceDescriptor_gpudevice, Variable, Parameter, ConstantFloat, ConstantDouble, Constant, Placeholder, DataType_Float, DataType_Double, ParameterFloat, ParameterDouble
from cntk.graph import TensorOpsMixin
from .. import utils

FLOAT_32='float32'

def _sanitize_value(shape, value, dtype, device, is_param=False):
    np_dtype = utils.sanitize_dtype_numpy(dtype)
    cntk_dtype  = utils.sanitize_dtype_cntk(dtype)
    if value is None:
        if shape is None:
            raise ValueError('you need to specify at least shape or value')        
        if is_param:
            if not np.isscalar(shape):
                # cntk uses column major, thus we reverse the shape    
                shape = tuple(reversed(shape))
            # TODO: expose the initialization params
            ndav = NDArrayView.random_uniform_float(shape, -0.05, 0.05, 1, device)        
        else:
            ndav = utils.create_NDArrayView(shape, cntk_dtype, device)
    

    else:
        if not isinstance(value, np.ndarray) or value.dtype!=np_dtype:
            value = np.asarray(value, dtype=np_dtype)

        #TODO: check whether this copy operation from cpu to gpu is not needed
        if device != DeviceDescriptor_cpudevice():
            ndav_cpu = utils.create_NDArrayView_from_NumPy(value)
            ndav = utils.create_NDArrayView(value.shape, data_type=cntk_dtype, dev=device)
            ndav.copy_from(ndav_cpu)
        else:
            ndav = utils.create_NDArrayView_from_NumPy(value, device)

    return ndav

class Variable(Variable, TensorOpsMixin):
    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False, name=''):
        if not np.isscalar(shape):
            # cntk uses column major, thus we reverse the shape    
            shape = tuple(reversed(shape))        

        if data_type is None:            
            data_type = FLOAT_32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse, dtype, needs_gradient, name)

class Parameter(Parameter, TensorOpsMixin):
    def __init__(self, shape=None, value=None, data_type=None, device=None, name=''):
        
        if data_type is None:
            if not isinstance(value, np.ndarray):        
                data_type = FLOAT_32
            else:
                data_type = str(value.dtype)        

        ndav = _sanitize_value(shape, value, data_type, device, True)
        super(Parameter, self).__init__(ndav, name)

# TODO: make this part of the above constructor
def parameter_from_scalar(shape=None, value=None, data_type=None, device=None, name=''):
    if not device:
        device = DeviceDescriptor_cpudevice()            

    if data_type is None:
        if not isinstance(value, np.ndarray):        
            data_type = 'float32'
        else:
            data_type = str(value.dtype)     

    dtype = utils.sanitize_dtype_cntk(data_type)

    if not shape:
        shape = ()
    if dtype == DataType_Float:
        return ParameterFloat(shape, value, device, name)
    elif dtype == DataType_Double:
        return ParameterDouble(shape, value, device, name)
    raise ValueError('unrecognized data_type: %s', dtype)

# TODO: make this part of the above constructor
class Constant(Constant, TensorOpsMixin):
    def __init__(self, shape=None, value=None, data_type=None, device=None, name=''):

        if data_type is None:
            if not isinstance(value, np.ndarray):        
                data_type = FLOAT_32
            else:
                data_type = str(value.dtype)     

        if not device:
            device = DeviceDescriptor_cpudevice()            

        ndav = _sanitize_value(shape, value, data_type, device)
        super(Constant, self).__init__(ndav, name)

def constant_from_scalar(shape=None, value=None, data_type=None, device=None, name=''):
    if not device:
        device = DeviceDescriptor_cpudevice()            

    if data_type is None:
        if not isinstance(value, np.ndarray):        
            data_type = 'float32'
        else:
            data_type = str(value.dtype)     

    dtype = utils.sanitize_dtype_cntk(data_type)

    if not shape:
        shape = ()
    if dtype == DataType_Float:
        return ConstantFloat(shape, value, device, name)
    elif dtype == DataType_Double:
        return ConstantDouble(shape, value, device, name)
    raise ValueError('unrecognized data_type: %s', dtype)

class Placeholder(Placeholder, TensorOpsMixin):    
    def __init__(self, shape=None, name=''):

        if not np.isscalar(shape):
            # cntk uses column major, thus we reverse the shape    
            shape = tuple(reversed(shape))

        super(Placeholder, self).__init__(shape, name)
        
