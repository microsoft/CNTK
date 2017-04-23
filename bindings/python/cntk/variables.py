import numpy as np

from . import cntk_py
from .core import NDArrayView
from .device import DeviceDescriptor, use_default_device
from .tensor import TensorOpsMixin
from .default_options import get_default_override, default_override_or
from .internal import typemap, sanitize_precision, sanitize_value, \
        sanitize_shape, sanitize_dtype_cntk

class Record(dict):
    '''
    Easy construction of a record (=immutable singleton class) from keyword arguments.
    e.g. r = Record(x = 13, y = 42) ; x = r.x

    Args:
        kwargs: keyword arguments to turn into the record members

    Returns:
        A singleton class instance that has all passed kw args as immutable class members.
    '''
    def __init__(self, **args_dict):
        super(Record, self).__init__(args_dict)
        self.__dict__.update(args_dict)
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("record has no attribute '{}'".format(key))
        return self[key]

    def __setattr__(self, key, value):
        raise AttributeError('record is immutable')
    def updated_with(self, **kwargs):
        '''
        Create a new Record from an existing one with members modified or added.
        e.g. r = Record(x = 13) ; print(r.x) ; r2 = r.updated_with(x = 42) ; print(r2.x)
    
        Args:
            kwargs: keyword arguments to turn into the record members
    
        Returns:
            A singleton class instance that has all passed kw args as immutable class members.
        '''
        d = dict(**self)   # make it mutable
        d.update(kwargs)   # merge the new items
        return Record(**d) # lock it up again


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

    class Type(Record):
        '''
        Describes a Variable's type; that is, all arguments to instantiate a Placeholder or Input.
        These are meant to be passed to update_signature.
        All are optional, meaning unspecified.
        '''
        def __init__(self, shape=None, dtype=None, needs_gradient=None, is_sparse=None, dynamic_axes=None):
            r = dict()
            if shape is not None:
                r['shape'] = shape
            if dtype is not None:
                r['dtype'] = dtype
            if needs_gradient is not None:
                r['needs_gradient'] = needs_gradient
            if is_sparse is not None:
                r['is_sparse'] = is_sparse
            if dynamic_axes is not None:
                r['dynamic_axes'] = dynamic_axes
            super(Variable.Type, self).__init__(**r)

        def __str__(self):
            '''
            Stringifies the Type record back to Python 3 syntax per layers.typing.
            '''
            # base type
            unknown_shape = (-2,)
            shape     = getattr(self, 'shape', unknown_shape)
            is_sparse = getattr(self, 'is_sparse', False)
            axes      = getattr(self, 'dynamic_axes', ())
            has_axes = len(axes) > 0 # it's a tuple of Axis
            if is_sparse and not has_axes:
                raise TypeError('Type: cannot be sparse and not have an axis')
            if shape == unknown_shape:  #.is_unknown():  # TODO: how to do this right?
                s = 'tensor'
            elif shape == ():
                s = 'float'
            else:
                s = 'Tensor[' + ','.join(str(dim) for dim in shape) + ']'
                if is_sparse:
                    s = "Sparse" + s
                elif not has_axes:
                    s = "Parameter" + s
            # axis
            if has_axes:
                for axis in reversed(axes):
                    if axis.name == 'defaultBatchAxis':  # axis == Axis.default_batch_axis():  --TODO: how to do this right?
                        continue
                    if axis.name == 'defaultDynamicAxis' or axis.name == 'UnknownAxes': # TODO: how to do this right?
                        t = 'Sequence'
                    else:
                        t = 'SequenceOver[' + axis.name + ']'
                    s = t + '[' + s + ']'
            # We do not return dtype or needs_gradient. dtype is mostly redundant, and needs_gradient is not really part of the type.
            return s

    @property
    def type(self):
        '''
        The complete type of the data represented by this Variable as a single Variable.Type instance.
        '''
        return Variable.Type(shape=self.shape, dtype=self.dtype, needs_gradient=self.needs_gradient, is_sparse=self.is_sparse, dynamic_axes=self.dynamic_axes)



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
    '''__init__(self, shape=None, init=None, dtype=np.float32, device=None, name='')

    A trainable parameter. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that can be modified by a training
    procedure.

    Example:
         >>> p = C.Parameter((13,42,7), init=C.glorot_uniform())
         >>> p.shape
             (13, 42, 7)

         >>> # example with inferred dimensions
         >>> W = C.Parameter((C.InferredDimension, 42), init=C.glorot_uniform())
         >>> W.shape   # -1 indicates dimension yet to be inferred
             (-1, 42)
         >>> x = C.input(13)
         >>> y = C.times(x, W)  # times operation now knows that the input dimension of W must be 13
         >>> W.shape          # hence, the shape has been updated
             (13, 42)

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
    def __init__(self, shape=None, init=None, dtype=default_override_or(np.float32),
                 device=None, name=''):
        if not device:
            device = use_default_device()

        if init is None:
            init = 0

        pure = get_default_override(None, pure=default_override_or(False))
        if pure:
            raise TypeError('parameters cannot be created inside a @Function def')

        from _cntk_py import constant_initializer
        if np.isscalar(init):
            init = constant_initializer(init)
            if not shape:
                shape = ()

        dtype = get_default_override(Parameter, dtype=dtype)
        if dtype is not None:
            if isinstance(init, np.ndarray) and dtype != init.dtype:
                init = np.array(init, dtype=dtype)
        else:
            if isinstance(init, np.ndarray):
                dtype = init.dtype
            else:
                dtype = np.float32

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
    '''__init__(self, value=None, shape=None, dtype=np.float32, device=None, name='')

    A constant value. It can be a scalar, vector, matrix, or tensor
    of floating point numbers that cannot be modified.

    A Constant is a :class:`~cntk.variables.Variable` and therefore inherits all its methods.

    Example:
         >>> c = C.Constant(1, (2,3))
         >>> c.value
             array([[ 1.,  1.,  1.],
                    [ 1.,  1.,  1.]], dtype=float32)

    Args:
       value (`np.ndarray` or `list` or `float` or `int`): Initial value.
        BUGBUG: Document initializers
       dtype (`np.float32` or `np.float64`): data type to store the values as.
       device (:class:`~cntk.device.DeviceDescriptor`): the device on which the values should reside.
       name (`str`): an optional name for this constant.
    '''
    def __init__(self, value=None, shape=None, dtype=default_override_or(np.float32), device=None, name=''):

        if not device:
            device = use_default_device()

        if (np.isscalar(value) or isinstance(value, np.ndarray)) and not shape:
            shape = ()

        dtype = get_default_override(Constant, dtype=dtype)
        if dtype is not None:
            if isinstance(value, np.ndarray) and dtype != value.dtype:
                value = np.array(value, dtype=dtype)
        else:
            if isinstance(value, np.ndarray):
                dtype = value.dtype
            else:
                dtype = np.float32

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

