import numpy as np
from cntk import cntk_py, NDArrayView
from cntk.device import DeviceDescriptor
from ..tensor import TensorOpsMixin
from ..utils import typemap, sanitize_precision, sanitize_value, \
        sanitize_shape, sanitize_dtype_cntk

class VariableMixin(object):
    '''
    Standard properties for :class:`Variable` and its derived classes
    :class:`Parameter` and :class:`Constant`.
    '''
    @property
    @typemap
    def dynamic_axes(self):
        '''
        The dynamic axes of this variable.
        '''
        return super(VariableMixin, self).dynamic_axes()

    @property
    def dtype(self):
        '''
        The NumPy type of this variable.
        '''
        return sanitize_precision(self.get_data_type())

    @property
    def is_constant(self):
        '''
        Whether this variable is a constant.
        '''
        return super(VariableMixin, self).is_constant()

    @property
    def is_input(self):
        '''
        Whether this variable is an input.
        '''
        return super(VariableMixin, self).is_input()

    @property
    def is_output(self):
        '''
        Whether this variable is an output.
        '''
        return super(VariableMixin, self).is_output()

    @property
    def is_parameter(self):
        '''
        Whether this variable is a parameter.
        '''
        return super(VariableMixin, self).is_parameter()

    @property
    def is_placeholder(self):
        '''
        Whether this variable is a placeholder.
        '''
        return super(VariableMixin, self).is_placeholder()

    @property
    def is_sparse(self):
        '''
        Whether this variable is sparse.
        '''
        return super(VariableMixin, self).is_sparse()

    @property
    def name(self):
        '''
        The name of this variable.
        '''
        return super(VariableMixin, self).name()

    @property
    def needs_gradient(self):
        '''
        Whether this variable needs gradients.
        '''
        return super(VariableMixin, self).needs_gradient()

    @property
    @typemap
    def owner(self):
        '''
        The function this variable is an output of.
        '''
        if self.is_output == False:
            raise RuntimeError('called owner() on a variable that is not an output variable')
        return super(VariableMixin, self).owner()

    @property
    def shape(self):
        '''
        The shape of this variable as a tuple.
        '''
        return super(VariableMixin, self).shape().dimensions()

    @property
    def uid(self):
        '''
        The internally generated unique name of the variable.
        '''
        return super(VariableMixin, self).uid()



class Variable(VariableMixin, TensorOpsMixin, cntk_py.Variable):
    '''Variable(shape=None, dtype=None, needs_gradient=False, is_sparse=False, dynamic_axes=[Axis.default_batch_axis(), Axis.default_dynamic_axis()], name='')

    Denotes a symbolic entity corresponding to the inputs and outputs of a Function.

    Args:
       shape (`tuple`): the shape of this variable.
       dtype (`np.float32 or np.float64`): data type of the values that will be bound to this variable.
        Default is np.float32
       needs_gradient (`bool`): if set to True any expression that contains this variable
        will also be differentiated with respect to this variable.
       is_sparse(`bool`): whether this is a sparse or dense input (or output)
       dynamic_axes(`list` of :class:`~cntk.axis.Axis`): the dynamic axes of this variable. These
        express dimensions that can vary across examples or minibatches.
       name(`str`): an optional name for this parameter.
    '''
    def __init__(self, shape=None, dtype=None, needs_gradient=False, is_sparse=False,
                 dynamic_axes=[cntk_py.Axis.default_batch_axis(), cntk_py.Axis.default_dynamic_axis()], name=''):
        shape = sanitize_shape(shape)

        if dtype is None:
            dtype = np.float32
        dtype = sanitize_dtype_cntk(dtype)

        dynamic_axes = sanitize_dynamic_axes(dynamic_axes)

        super(Variable, self).__init__(shape, is_sparse, dtype, needs_gradient, name, dynamic_axes)

    @typemap
    def as_parameter(self):
        '''
        Converts this instance into a :class:`Parameter`
        '''
        if not self.is_parameter:
            raise TypeError('cannot be converted into a Parameter')

        return cntk_py.Parameter(self)

    @typemap
    def as_constant(self):
        '''
        Converts this instance into a :class:`Constant`
        '''
        if not self.is_constant:
            raise TypeError('cannot be converted into a Constant')

        return cntk_py.Constant(self)


class Parameter(VariableMixin, TensorOpsMixin, cntk_py.Parameter):
    '''
    A trainable parameter. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that can be modified by a training
    procedure.

    Args:
       shape (`tuple`): the shape of the tensor holding the parameters
       init (value (`np.ndarray`, `list`, `float`, `int`) or
        :class:`~cntk.initializer`: Initial value.
        If a numpy array is specified the shape argument is ignored and
        the tensor gets the shape of this argument. Alternatively, an
        initializer from :class:`~cntk.initializer` can be specified.
       dtype (`np.float32` or `np.float64`): data type of the values stored.
       device (:class:`~cntk.device.DeviceDescriptor`): the device on which the values should reside.
       name (`str`): an optional name for this parameter

    Parameters are Variables and therefore they inherit all their methods.
    '''
    def __init__(self, shape=None, init=None, dtype=None,
                 device=None, name=''):

        if dtype is None:
            if isinstance(init, np.ndarray):
                dtype = init.dtype
            else:
                dtype = np.float32

        if init is None:
            init = 0

        if isinstance(init, (np.ndarray, list, float, int)):
            ndav = sanitize_value(shape, init, dtype, device)
            super(Parameter, self).__init__(ndav, name)
        else:
            shape = sanitize_shape(shape)
            cntk_dtype = sanitize_dtype_cntk(dtype)
            super(Parameter, self).__init__(shape, cntk_dtype, init,
                    device, name)

    @property
    def value(self):
        '''
        NumPy array of the value
        '''
        return super(Parameter, self).value().to_ndarray()

    @value.setter
    def value(self, val):
        if isinstance(val, np.ndarray):
            ndarray = NDArrayView.from_dense(val.astype(self.dtype))
            super(Parameter, self).set_value(ndarray)
        elif isinstance(val, cntk_py.NDArrayView):
            super(Parameter, self).set_value(val)
        else:
            raise TypeError("Unsupported value type: %s", type(val))


class Constant(VariableMixin, TensorOpsMixin, cntk_py.Constant):
    '''
    A constant value. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that cannot be modified.

    A Constant is a :class:`~cntk.ops.Variable` and therefore inherits all its methods.

    Args:
       value (`np.ndarray` or `list` or `float` or `int`): Initial value.
        BUGBUG: Document initializers
       dtype (`np.float32` or `np.float64`): data type to store the values as.
       device (:class:`~cntk.device.DeviceDescriptor`): the device on which the values should reside.
       name (`str`): an optional name for this constant.
    '''
    def __init__(self, value=None, shape=None, dtype=None, device=None, name=''):

        if dtype is None:
            if isinstance(value, np.ndarray):
                dtype = value.dtype
            else:
                dtype = np.float32

        if device is None:
            device = DeviceDescriptor.use_default_device()

        if np.isscalar(value):
            super(Constant, self).__init__(sanitize_shape(shape),
                    sanitize_dtype_cntk(dtype), value, device, name)
        else:
            ndav = sanitize_value(shape, value, dtype, device)
            super(Constant, self).__init__(ndav, name)

    @property
    def value(self):
        '''
        NumPy array of the value
        '''
        return super(Constant, self).value().to_ndarray()

