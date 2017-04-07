# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import warnings
from scipy import sparse

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
    def asarray(self):
        '''
        Converts the instance's data to a NumPy array.
        '''
        import cntk
        result = None
        if isinstance(self, cntk.Constant):
            ndav = super(cntk.Constant, self).value()
            is_sparse = ndav.is_sparse()
        elif isinstance(self, cntk.Parameter):
            ndav = super(cntk.Parameter, self).value()
            is_sparse = ndav.is_sparse()
        elif isinstance(self, (cntk.cntk_py.Constant, cntk.cntk_py.Parameter)):
            ndav = self.value()
            is_sparse = ndav.is_sparse()

        elif isinstance(self, (cntk.cntk_py.NDArrayView, cntk.cntk_py.NDMask)):
            ndav = self
            if isinstance(self, cntk.NDArrayView):
                is_sparse = ndav.is_sparse
            elif isinstance(self, cntk.cntk_py.NDArrayView):
                is_sparse = ndav.is_sparse()
            else:
                is_sparse = False

        # Value and MinibatchData have a mask, which means that we need the
        # corresponding Variable to do the proper conversion. For easy
        # discoverability, we nevertheless add asarray() to those classes as
        # well, but issue a warning.
        elif isinstance(self, cntk.cntk_py.Value) or isinstance(self, cntk.cntk_py.MinibatchData):

            if isinstance(self, cntk.cntk_py.MinibatchData):
                value = self.data
            else:
                value = self

            if isinstance(value, cntk.Value):
                is_sparse = value.is_sparse
                has_mask = super(cntk.Value, value).mask() is not None
                ndav = value.data
            else:
                is_sparse = value.is_sparse()
                has_mask = value.mask() is not None
                ndav = value.data()

            if has_mask:
                warnings.warn('asarray() will ignore the mask information. '
                              'Please use as_sequences() to do the proper '
                              'conversion.')

        if is_sparse:
            from cntk.internal.sanitize import _sparse_to_dense_network_cache

            device = self.device
            if callable(device):
                device = device()

            network = _sparse_to_dense_network_cache(ndav.shape[1:], False,
                    device)
            warnings.warn('converting Value object to CSR format might be slow')
 
            dense_data = network.eval(self, device=device)

            def to_csr(dense_data):
                if len(dense_data.shape) > 2:
                    raise ValueError('Cannot convert a sparse NDArrayView or Value object '
                                     'with shape %s of rank > 2 to a scipy.csr matrix.' % str(dense_data.shape))
                return sparse.csr_matrix(dense_data)

            if isinstance(dense_data, list):
                result = [to_csr(d) for d in dense_data]
            else:
                result = to_csr(dense_data)

        else:
            result = ndav.to_ndarray()

        return result

def _add_asarray(klass):
    member_name = 'asarray'

    if getattr(klass, member_name, None):
        raise ValueError('class "%s" has already an attribute "%s"' %
                         (klass, member_name))

    setattr(klass, member_name, ArrayMixin.__dict__[member_name])
