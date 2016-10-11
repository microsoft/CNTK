# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


class TensorOpsMixin(object):

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

        setattr(klass, overload_name, getattr(TensorOpsMixin, overload_name))


class EvalMixin(object):

    def eval(self, arguments=None, precision='float', device=None):
        '''
        Evaluate the node using the specified `arguments` as input.

        Args:
            arguments (`dict` or `list`): 
              * map from input variables to the data
              * list of inputs in the order that the function expects or 
              Data should be either NumPy arrays or a `:class:cntk.io.MinibatchData` instance
            precision (`str` or `np.float32` or `np.float64`): precision, if string
             it can be one of 'float' 'float32, 'double', 'float64', or `None`
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
        '''
        from .utils import eval as utils_eval
        if device is None:
            from . import DeviceDescriptor
            device = DeviceDescriptor.use_default_device()

        if len(self.outputs()) != 1:
            raise ValueError(
                'only operators with exactly one output can be evaluated')

        if arguments is None:
            arguments = {}

        result, _ = utils_eval(self, precision, device, arguments, False)
        return result.popitem()[1]


def _add_eval(klass):
    overload_name = 'eval'

    if getattr(klass, overload_name, None):
        raise ValueError('class "%s" already has operator overload "%s"' %
                         (klass, overload_name))

    setattr(klass, overload_name, getattr(EvalMixin, overload_name))
