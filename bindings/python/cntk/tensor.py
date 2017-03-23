# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


class TensorOpsMixin(object):
    '''
    This class defines math overloads so that CNTK nodes can be written in math
    expressions.
    '''

    # operator overload for (+) where self is the left operand
    def __add__(self, other):
        from . import ops
        return ops.plus(self, other)

    # operator overload for (+) where self is the right operand
    def __radd__(self, other):
        from . import ops
        return ops.plus(other, self)

    # operator overload for (-) where self is the left operand
    def __sub__(self, other):
        from . import ops
        return ops.minus(self, other)

    # operator overload for (-) where self is the right operand
    def __rsub__(self, other):
        from . import ops
        return ops.minus(other, self)

    # operator overload for (*) where self is the left operand
    def __mul__(self, other):
        from . import ops
        return ops.element_times(self, other)

    # operator overload for (*) where self is the right operand
    def __rmul__(self, other):
        from . import ops
        return ops.element_times(other, self)

    # operator overload for (@) where self is the left operand
    def __matmul__(self, other):
        # NOTE supported in Python 3.5
        from . import ops
        return ops.times(self, other)

    # operator overload for (@) where self is the right operand
    def __rmatmul__(self, other):
        # NOTE supported in Python 3.5
        from . import ops
        return ops.times(other, self)

    # operator overload for (/) where self is the left operand
    def __truediv__(self, other):
        from . import ops
        self.__div__ = self.__truediv__
        return ops.element_divide(self, other)

    # operator overload for (/) where self is the right operand
    def __rtruediv__(self, other):
        from . import ops
        self.__rdiv__ = self.__rtruediv__
        return ops.element_divide(other, self)

    # Python2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __abs__(self):
        from . import ops
        return ops.abs(self)

    def __neg__(self):
        from . import ops
        return ops.negate(self)

    # TODO __xor__, __rxor__, __pow__, __rpow__,  __invert__

    # Comparison operators are not exposed yet, because of __eq__ being
    # required to allow comparison of Variables on C++ so that we can say
    # 'for var in variables'.
    # __lt__, __le__, __gt__, __ge__, __and__, __rand__, __or__, __ror__,

    def __getitem__(self, arg):
        '''
        Slicing of a Variable. E.g. var[2:3] will translate into slice(var, axis=0, begin_index=2, end_index=3)
        '''
        from . import ops

        # int or slice: normalize into a tuple of int or tuple of slice
        if not isinstance(arg, tuple): 
            arg = (arg,)
        r = self
        axis0 = 0

        for axis, s in enumerate(arg):
            if s is Ellipsis: # ellipsis means index relative to end after this point
                axis0 = -len(arg)
                continue
            if isinstance(s, int): # int: normalize into a slice
                s = slice(s, s+1)

            if isinstance(s, slice):
                if s.step is not None and s.step != 1:
                    # TODO: This is not hard to implement in SliceNode.
                    raise ValueError("slicing with a step other than 1 is "
                                     "currently not supported")
                # implement as a CNTK slice() operation
                begin = s.start or 0
                end   = s.stop  or 0
                if begin != 0 or end != 0:
                    r = ops.slice(r, axis=axis + axis0, begin_index=begin, end_index=end)
            elif isinstance(s, (tuple, list)):
                # Select multiple elements from the same dimension. This is
                # different from NumPy's advanced indexing, since we just go
                # axis by axis from left to right and don't do any
                # broadcasting.

                slice_accum = []
                for idx in s:
                    if not isinstance(idx, int):
                        raise IndexError(
                              'indices have to be of type int and not "%s"' %
                               type(idx))
                    slice_accum.append(ops.slice(r, axis=axis,
                                                 begin_index=idx,
                                                 end_index=idx + 1))
                if len(slice_accum) > 1:
                    r = ops.splice(*slice_accum, axis=axis)
                else:
                    r = slice_accum[0]
            else:
                raise IndexError(
                    'type "%s" is not supported as index' % type(s))

        return r


AVAILABLE_TENSOR_OPS = ['abs', 'add', 'div', 'getitem', 'matmul', 'mul',
                        'radd', 'rdiv', 'rmatmul', 'rmul', 'rsub', 'rtruediv',
                        'sub', 'truediv', 'neg']


def _add_tensor_ops(klass):
    for op_name in AVAILABLE_TENSOR_OPS:
        overload_name = '__%s__' % op_name

        if getattr(klass, overload_name, None):
            raise ValueError('class "%s" already has operator overload "%s"' %
                             (klass, overload_name))

        setattr(klass, overload_name, TensorOpsMixin.__dict__[overload_name])


class ArrayMixin(object):
    @property
    def __array_interface__(self):
        import cntk
        if isinstance(self, (cntk.cntk_py.Constant, cntk.cntk_py.Parameter, cntk.cntk_py.MinibatchData)):
            np_array = self.value
        elif isinstance(self, cntk.core.Value):
            np_array = self.data.to_ndarray()
        elif isinstance(self, cntk.cntk_py.Value):
            np_array = self.data().to_ndarray()
        elif isinstance(self, (cntk.cntk_py.NDArrayView, cntk.cntk_py.NDMask)):
            np_array = self.to_ndarray()
        else:
            # Ideally an exception would be raised here, but getattr would swallow it
            # so we return None
            return None

        interface_copy = np_array.__array_interface__

        # for np arrays (other than 0-d arrays) data entry in __array_interface__ dict
        # must be replaced with data member of array
        if len(np_array.shape):
            interface_copy["data"] = np_array.data
        else:
            # save a reference to np_array so that it does not disappear
            self.np_array = np_array

        return interface_copy

def _add_array_interface(klass):
    array_interface_name = '__array_interface__'

    if getattr(klass, array_interface_name, None):
        raise ValueError('class "%s" has already an attribute "%s"' %
                         (klass, array_interface_name))

    setattr(klass, array_interface_name, getattr(ArrayMixin, array_interface_name))
