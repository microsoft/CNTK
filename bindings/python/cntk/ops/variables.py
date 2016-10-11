import numpy as np
from cntk import cntk_py
#from cntk.cntk_py import NDArrayView, DeviceDescriptor, Variable, Parameter, ConstantFloat, ConstantDouble, Constant, DataType_Float, DataType_Double, ParameterFloat, ParameterDouble, Axis
#from cntk import DATATYPE
from cntk.tensor import TensorOpsMixin
from cntk import utils

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

class Variable(TensorOpsMixin, cntk_py.Variable):
    '''
    Denotes a symbolic entity corresponding to the inputs and outputs of a Function.

    Args:
       shape (`tuple`): the shape of this variable.
       data_type (`np.float32 or np.float64`): data type of the values that will be bound to this variable.
        Default is np.float32
       needs_gradient (`bool`): if set to True any expression that contains this variable
        will also be differentiated with respect to this variable.
       is_sparse(`bool`): whether this is a sparse or dense input (or output)
       dynamic_axes(`list` of `cntk.Axis`): the dynamic axes of this variable. These
        express dimensions that can vary across examples or minibatches.
       name(`str`): an optional name for this parameter.

    :ivar dynamic_axes: the dynamic axes of this variable
    :ivar dtype: data type of the values that will be bound to this variable
    :ivar is_sparse: whether this variable is sparse
    :ivar is_input: whether this variable is an input in the computational network
    :ivar is_output: whether this variable is an output of the computational network
    :ivar is_placeholder: whether this variable is a placeholder
    :ivar name: the name of this variable
    :ivar needs_gradient: whether the gradient will be computed for this variable
    :ivar owner: the function object that is the owner of this variable
    :ivar shape: the shape of this variable
    :ivar uid: the internally generated unique name of this variable
    '''
    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False,
                 dynamic_axes=[cntk_py.Axis.default_dynamic_axis(), cntk_py.Axis.default_batch_axis()], name=''):
        shape = utils.sanitize_shape(shape)

        if data_type is None:
            data_type = FLOAT_32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse,
                                       dtype, needs_gradient, name, dynamic_axes)
                                       

    #@typemap
    def dynamic_axes(self):
        '''
        Returns the dynamic axes of this variable

        Returns:
            `:class:cntk.Axis`
        '''
        return super(cntk_py.Variable, self).dynamic_axes()

class Parameter(TensorOpsMixin, cntk_py.Parameter):
    '''
    A trainable parameter. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that can be modified by a training
    procedure.

    Args:
       shape (`tuple`): the shape of the tensor holding the parameters
       init (`np.ndarray` or `list` or `float` or `int`): Initial value.
        If a numpy array is specified the shape argument is ignored and
        the tensor gets the shape of this argument.
       data_type (`np.float32 or np.float64`): data type of the values stored.
       device (`dev`): the device on which the values should reside.
       name (`str`): an optional name for this parameter

    Parameters are Variables and therefore they inherit all their attributes.
    '''
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

class Constant(TensorOpsMixin, cntk_py.Constant):
    '''
    A constant value. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that cannot be modified.

    Args:
       value (`np.ndarray` or `list` or `float` or `int`): Initial value.
       data_type (`np.float32 or np.float64`): data type to store the values as.
       device (`dev`): the device on which the values should reside.
       name (`str`): an optional name for this constant.

    Constants are Variables and therefore they inherit all their attributes.

    :ivar value: the value of the constant
    '''
    def __init__(self, value, data_type=None, device=None, name=''):

        if data_type is None:
            data_type = str(value.dtype)

        ndav = _sanitize_value(value.shape, value, data_type, device)
        super(Constant, self).__init__(ndav, name)

        self.value = super().value()
    #scalar
