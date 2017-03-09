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

    def __getitem__(self, key):
        from . import ops
        if isinstance(key, int):
            # Case 1: e.g. data[3] -> key=3
            return ops.slice(self, 0, key, key + 1)

        elif isinstance(key, slice):
            # Case 2: e.g. data[2:4] -> key will be a slice object
            if key.step is not None:
                raise TypeError('step argument is not supported')
            if not isinstance(key.stop, int):
                raise TypeError(
                    'end index has to be of type int, not "%s"' % type(key.stop))

            if isinstance(key.start, int):
                if key.stop <= key.start:
                    raise ValueError(
                        'end index has to be greater than start index')
            return ops.slice(self, 0, key.start or 0, key.stop or 0)

        elif isinstance(key, (tuple, list)):
            # Case 3: e.g. data[2:4,1:,1:7] -> key will be an iterable of ints
            # (case 1) or slices (case 2)
            # objects.
            # FIXME: we need to check that len(key) equals the node's rank
            node = self
            for ax_counter, so in enumerate(key):
                if isinstance(so, int):
                    # Proceed as case 1
                    node = ops.slice(node, ax_counter, so, so + 1)

                elif isinstance(so, slice):
                    # Proceed as case 2
                    if so.step is not None:
                        raise TypeError('step argument is not supported')
                    if isinstance(so.start, int) and isinstance(so.stop, int):
                        if so.stop <= so.start:
                            raise ValueError(
                                'end index has to be greater than start index')
                    if so.start is None and so.stop is None:
                        continue
                    node = ops.slice(node, ax_counter, so.start or 0, so.stop or 0)
                elif isinstance(so, list):
                    # Case 3b: e.g. data[[0],[2,3]] aka "advanced indexing" ->
                    # so = ([0], [2,3])
                    # In NumPy we would have another dimension, but since
                    # data[0].shape != data[[0]].shape == data[[[0]]].shape ==
                    # we decided to have all shapes like data[0] in this case
                    for idx in so:
                        if not isinstance(idx, int):
                            raise IndexError(
                                'indices have to be of type int and not "%s"' % type(idx))
                        node = ops.slice(node, ax_counter, idx, idx + 1)
                else:
                    raise IndexError(
                        'type "%s" is not supported as index' % type(so))

            return node
        else:
            raise TypeError(
                'index must be int or slice, not {}'.format(type(key).__name__))

AVAILABLE_TENSOR_OPS = ['abs', 'add', 'div', 'getitem', 'matmul', 'mul',
                        'radd', 'rdiv', 'rmatmul', 'rmul', 'rsub', 'rtruediv', 'sub',
                        'truediv', 'neg']


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
        try:
            # This checks for a MinibatchData object.
            np_array = self.value
        except AttributeError:
            try:
                # This checks for a Value object. Trying with self.to_ndarray first would lead to
                # a infinite recursion, since Value has a to_ndarray method
                np_array = self.data().to_ndarray()
            except AttributeError:
                try:
                    np_array = self.to_ndarray()
                except AttributeError:
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
