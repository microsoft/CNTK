import numpy as np
from cntk import cntk_py
#from cntk_py import NDArrayView, DeviceDescriptor, Variable, Parameter, ConstantFloat, ConstantDouble, Constant, DataType_Float, DataType_Double, ParameterFloat, ParameterDouble, Axis
#from cntk import DATATYPE
from cntk.tensor import TensorOpsMixin
from cntk import utils
from ..utils import typemap

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
    '''
    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False,
                 dynamic_axes=[cntk_py.Axis.default_dynamic_axis(), cntk_py.Axis.default_batch_axis()], name=''):
        shape = utils.sanitize_shape(shape)

        if data_type is None:
            data_type = FLOAT_32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse,
                                       dtype, needs_gradient, name, dynamic_axes)

    @typemap
    def dynamic_axes(self):
        '''
        Returns the dynamic axes of this variable

        Returns:
            `list`: list of `:class:cntk.Axis` that are the dynamic_axes of this Variable
        '''
        return super(Variable, self).dynamic_axes()

    @typemap
    def get_data_type(self):
        '''
        Returns the data type of the data that this Variable symbolically represents

        Returns:
            `DataType`: the data type of the data that this Variable symbolically represents
        '''
        return super(Variable, self).get_data_type()

    @typemap
    def is_constant(self):
        '''
        Returns True if this variable is a constant and False otherwise

        Returns:
            `bool`: True if this variable is a Constant and False otherwise
        '''
        return super(Variable, self).is_constant()

    @typemap
    def is_input(self):
        '''
        Returns True if this variable is an input and False otherwise

        Returns:
            `bool`: True if this variable is an input and False otherwise
        '''
        return super(Variable, self).is_input()

    @typemap
    def is_output(self):
        '''
        Returns True if this variable is an output and False otherwise

        Returns:
            `bool`: True if this variable is an output and False otherwise
        '''
        return super(Variable, self).is_output()

    @typemap
    def is_parameter(self):
        '''
        Returns True if this variable is a parameter and False otherwise

        Returns:
            `bool`: True if this variable is a parameter and False otherwise
        '''
        return super(Variable, self).is_parameter()

    @typemap
    def is_placeholder(self):
        '''
        Returns True if this variable is a placeholder and False otherwise

        Returns:
            `bool`: True if this variable is a placeholder and False otherwise
        '''
        return super(Variable, self).is_placeholder()

    @typemap
    def is_sparse(self):
        '''
        Returns True if this variable will be bound to sparse data and False otherwise

        Returns:
            `bool`: True if this variable will be bound to sparse data
        '''
        return super(Variable, self).is_sparse()

    # @typemap
    # def kind(self):
        # '''
        # kind
        

        # Returns:
            # `VariableKind`: text
        # '''
        # return super(Variable, self).kind()

    @typemap
    def name(self):
        '''
        Returns the name of this variable

        Returns:
            `str`: the name of this variable
        '''
        return super(Variable, self).name()

    @typemap
    def needs_gradient(self):
        '''
        Returns True if gradient computation is enabled for this variable and False otherwise.

        Returns:
            `bool`: True if gradient computation is enabled for this variable and False otherwise.
        '''
        return super(Variable, self).needs_gradient()

    @typemap
    def owner(self):
        '''
        Returns:
            `Function`: the Function object which 'this' variable is an ouptut of.
        '''
        if self.is_output() == False:
            raise RuntimeError('called owner() on a variable that is not an output variable')
        return super(Variable, self).owner()

    @typemap
    def shape(self):
        '''
        Returns:
            `NDShape`: the shape of the Variable
        '''
        return super(Variable, self).shape()

    @typemap
    def uid(self):
        '''
        Returns:
            `str`:  the internally generated unique name of the variable
        '''
        return super(Variable, self).uid()

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

    Parameters are Variables and therefore they inherit all their methods.
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

    @typemap
    def value(self):
        '''
        Returns:
            `NDArrayView`: the current value of the parameter.
        '''
        return super(Constant, self).value()

class Constant(TensorOpsMixin, cntk_py.Constant):
    '''
    A constant value. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that cannot be modified.

    Constants are :class:`cntk.ops.Variable`s and therefore they inherit all their methods.

    Args:
       value (`np.ndarray` or `list` or `float` or `int`): Initial value.
       data_type (`np.float32 or np.float64`): data type to store the values as.
       device (`dev`): the device on which the values should reside.
       name (`str`): an optional name for this constant.
    '''
    def __init__(self, shape=None, value=None, data_type=None, device=None, name=''):

        if data_type is None:
            if isinstance(value, np.ndarray):
                data_type = str(value.dtype)
            else:
                data_type = FLOAT_32
                
        ndav = _sanitize_value(shape, value, data_type, device)
        super(Constant, self).__init__(ndav, name)

    #TODO how to expose Scalar ?
    
    @typemap
    def value(self):
        '''
        Returns:
            `NDArrayView`: the value of the constant.
        '''
        return super(Constant, self).value()
