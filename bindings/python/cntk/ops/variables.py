import numpy as np
from cntk import cntk_py, utils
from ..tensor import TensorOpsMixin
from ..utils import typemap, sanitize_precision, sanitize_value


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
       dynamic_axes(`list` of :class:`cntk.axis.Axis`): the dynamic axes of this variable. These
        express dimensions that can vary across examples or minibatches.
       name(`str`): an optional name for this parameter.
    '''
    def __init__(self, shape=None, data_type=None, needs_gradient=False, is_sparse=False,
                 dynamic_axes=[cntk_py.Axis.default_dynamic_axis(), cntk_py.Axis.default_batch_axis()], name=''):
        shape = utils.sanitize_shape(shape)

        if data_type is None:
            data_type = np.float32
        dtype = utils.sanitize_dtype_cntk(data_type)

        super(Variable, self).__init__(shape, is_sparse,
                                       dtype, needs_gradient, name, dynamic_axes)

    @property
    @typemap
    def dynamic_axes(self):
        '''
        The dynamic axes of this variable.
        '''
        return super(Variable, self).dynamic_axes()

    @property
    def dtype(self):
        '''
        The NumPy type of this variable.
        '''
        return sanitize_precision(self.get_data_type())

    @property
    @typemap
    def is_constant(self):
        '''
        Whether this variable is a constant.
        '''
        return super(Variable, self).is_constant()

    @property
    @typemap
    def is_input(self):
        '''
        Whether this variable is an input.
        '''
        return super(Variable, self).is_input()

    @property
    @typemap
    def is_output(self):
        '''
        Whether this variable is an output.
        '''
        return super(Variable, self).is_output()

    @property
    @typemap
    def is_parameter(self):
        '''
        Whether this variable is a parameter.
        '''
        return super(Variable, self).is_parameter()

    @property
    @typemap
    def is_placeholder(self):
        '''
        Whether this variable is a placeholder.
        '''
        return super(Variable, self).is_placeholder()

    @property
    @typemap
    def is_sparse(self):
        '''
        Whether this variable is sparse.
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

    @property
    @typemap
    def name(self):
        '''
        The name of this variable.
        '''
        return super(Variable, self).name()

    @property
    @typemap
    def needs_gradient(self):
        '''
        Whether this variable needs gradients.
        '''
        return super(Variable, self).needs_gradient()

    @property
    @typemap
    def owner(self):
        '''
        The function this variable is an output of.
        '''
        if self.is_output == False:
            raise RuntimeError('called owner() on a variable that is not an output variable')
        return super(Variable, self).owner()

    @property
    def shape(self):
        '''
        The shape of this variable as a tuple.
        '''
        return super(Variable, self).shape().dimensions()

    @property
    @typemap
    def uid(self):
        '''
        The internally generated unique name of the variable.
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
            if isinstance(init, np.ndarray):
                data_type = str(init.dtype)
            else:
                data_type = np.float32

        if init is None:
            init = 0

        if isinstance(init, (np.ndarray, list, float, int)):
            ndav = sanitize_value(shape, init, data_type, device)
            super(Parameter, self).__init__(ndav, name)
        else:
            shape = utils.sanitize_shape(shape)
            data_type  = utils.sanitize_dtype_cntk(data_type)
            super(Parameter, self).__init__(shape, data_type, init,
                    device, name)

    @property
    @typemap
    def value(self):
        '''
        NumPy array of the value
        '''
        return super(Parameter, self).value().to_numpy()

    @property
    def shape(self):
        '''
        The shape of this parameter as a tuple.
        '''
        return super(Parameter, self).shape().dimensions()

    @property
    def dtype(self):
        '''
        The NumPy type of this variable.
        '''
        return sanitize_precision(self.get_data_type())

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
    def __init__(self, value=None, shape=None, data_type=None, device=None, name=''):

        if data_type is None:
            if isinstance(value, np.ndarray):
                data_type = str(value.dtype)
            else:
                data_type = np.float32
                
        ndav = sanitize_value(shape, value, data_type, device)

        super(Constant, self).__init__(ndav, name)

    #TODO how to expose Scalar ?
    @property
    @typemap
    def value(self):
        '''
        NumPy array of the value
        '''
        return super(Constant, self).value().to_numpy()

    @property
    def shape(self):
        '''
        The shape of this constant as tuple.
        '''
        return super(Constant, self).shape().dimensions()

    @property
    def dtype(self):
        '''
        The NumPy type of this variable.
        '''
        return sanitize_precision(self.get_data_type())

