# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import ops

class TensorOpsMixin():    

    # operator overload for (+) where self is the left operand
    def __add__(self, other):
        return ops.plus(self, other)

    # operator overload for (+) where self is the right operand
    def __radd__(self, other):
        return ops.plus(other, self)

    # operator overload for (-) where self is the left operand
    def __sub__(self, other):
        return ops.minus(self, other)

    # operator overload for (-) where self is the right operand
    def __rsub__(self, other):
        return ops.minus(other, self)

    # operator overload for (*) where self is the left operand
    def __mul__(self, other):
        return ops.element_times(self, other)

    # operator overload for (*) where self is the right operand
    def __rmul__(self, other):
        return ops.element_times(other, self)

    # operator overload for (@) where self is the left operand
    def __matmul__(self, other):
        # NOTE supported in Python 3.5
        return ops.times(self, other)

    # operator overload for (@) where self is the right operand
    def __rmatmul__(self, other):
        # NOTE supported in Python 3.5
        return ops.times(other, self)

    # operator overload for (\) where self is the left operand
    def __truediv__(self, other):
        self.__div__ = self.__truediv__
        return ops.element_divide(self, other)

    # operator overload for (\) where self is the right operand
    def __rtruediv__(self, other):
        self.__rdiv__ = self.__rtruediv__
        return ops.element_divide(other, self)

    # Python2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __abs__(self):
        return ops.abs(self)

    # TODO __lt__, __le__, __gt__, __ge__, __and__, __rand__, __or__, __ror__, __xor__, __rxor__, __pow__, __rpow__,  __invert__, __neg__

    def __getitem__(self, key):
        if isinstance(key, int):
            # Case 1: e.g. data[3] -> key=3
            return ops.slice(self, key, key+1, axis=0)

        elif isinstance(key, slice):
            # Case 2: e.g. data[2:4] -> key will be a slice object
            if key.step is not None:
                raise TypeError('step argument is not supported')
            if not isinstance(key.stop, int):
                raise TypeError('end index has to be of type int, not "%s"'%type(key.stop))

            if isinstance(key.start, int):
                if key.stop<=key.start:
                    raise ValueError('end index has to be greater than start index')
            return ops.slice(self, key.start or 0, key.stop or 0, axis=0)

        elif isinstance(key, (tuple, list)):
            # Case 3: e.g. data[2:4,1:,1:7] -> key will be an iterable of ints
            # (case 1) or slices (case 2)
            # objects.
            # FIXME: we need to check that len(key) equals the node's rank
            node = self
            for ax_counter, so in enumerate(key):
                if isinstance(so, int):
                    # Proceed as case 1
                    node = ops.slice(node, so, so+1, axis=ax_counter)

                elif isinstance(so, slice):
                    # Proceed as case 2
                    if so.step is not None:
                        raise TypeError('step argument is not supported')
                    if isinstance(so.start, int) and isinstance(so.stop, int):
                        if so.stop<=so.start:
                            raise ValueError('end index has to be greater than start index')
                    if so.start is None and so.stop is None:
                        continue
                    node = ops.slice(node, so.start or 0, so.stop or 0, axis=ax_counter)
                elif isinstance(so, list):
                    # Case 3b: e.g. data[[0],[2,3]] aka "advanced indexing" ->
                    # so = ([0], [2,3])
                    # In NumPy we would have another dimension, but since
                    # data[0].shape != data[[0]].shape == data[[[0]]].shape ==
                    # we decided to have all shapes like data[0] in this case
                    for idx in so:
                        if not isinstance(idx, int):
                            raise IndexError('indices have to be of type int and not "%s"'%type(idx))
                        node = ops.slice(node, idx, idx+1, axis=ax_counter)
                else:
                    raise IndexError('type "%s" is not supported as index'%type(so))

            return node
        else:
            raise TypeError('index must be int or slice, not {}'.format(type(key).__name__))
