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

# --- experimental direct-mode support for simple NDArrayView operations ---

# TODO: move this to contrib/tensor_ops.py

from cntk.core import NDArrayView

class NDArrayViewOpsMixin(object):
    '''
    This class defines math overloads so that CNTK NDArrayViews can be operated on in math
    expressions.
    '''

    @staticmethod
    def _num_op(out, *args):
        if out:
            out.numeric_operation_in_place(0, *(args + (24,))) # 24 = ElementWiseOperator.opSum, a dummy here
            res = out # the return value is out but as the underlying raw class type
        else:
            res = NDArrayView.numeric_operation(*args)
            res.__class__ = args[0][0].__class__ # upgrade the type
        return res
    @staticmethod
    def _mat_prod(out, *args):
        if out:
            out.matrix_product_in_place(0, *args)
            res = out
        else:
            res = NDArrayView.matrix_product(*(args + (1,))) # last arg is output rank
            res.__class__ = args[1].__class__
        return res

    # infix operators
    def __add__(self, other, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self, other], 1.0, 24) # 24 = ElementWiseOperator.opSum
    def __sub__(self, other, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self, other], 1.0, 25) # 25 = ElementWiseOperator.opDifference
    def __mul__(self, other, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self, other], 1.0, 26) # 26 = ElementWiseOperator.opElementwiseProduct
    def greater(self, other, out=None):   # __gt__ somehow fails, already exists in cntk_py version
        return NDArrayViewOpsMixin._num_op(out, [self, other], 1.0, 35) # 35 = ElementWiseOperator.opGreater

    # so far these make no sense since we don't type-cast anyway
    __radd__ = __add__
    __rmul__ = __mul__

    # in-place variants
    def __iadd__(self, other):
        self.numeric_operation_in_place(1.0, [other], 1.0, 2, 24) # 2 = ElementWiseOperator.opCopy
        return self
    def __isub__(self, other): # realized as an in-place add-to with alpha=-1
        self.numeric_operation_in_place(1.0, [other], -1.0, 2, 24) # 2 = ElementWiseOperator.opCopy
        return self

    def __matmul__(self, other, out=None):
        shapeA = self.shape
        shapeB = other.shape
        if len(self.shape) == 0: # TODO: allow for scalar zero (initial_state)
            self1 = NDArrayView(shape=(other.shape[0]), data_type=other.dtype, device=other.device) # reduce to scalar
            # BUGBUG: How to get the precision in the right way?
            # TODO: test case
            self1.numeric_operation_in_place(0.0, [self], 1.0, 2, 24) # 2 = ElementWiseOperator.opCopy
            self = self1
        res = NDArrayViewOpsMixin._mat_prod(out, False, other, False, self, False, 1.0) # note: shapes are swapped, so we swap the order as well
        shapeC = res.shape
        return res
    dot = __matmul__
    def dot_transpose(self, other, out=None): # other gets transposed
        return NDArrayViewOpsMixin._mat_prod(out, False, other, True, self, False, 1.0) # note: shapes are swapped, so we swap the order as well
    def transpose_dot(self, other, out=None): # self gets transposed
        return NDArrayViewOpsMixin._mat_prod(out, False, other, False, self, True, 1.0) # note: shapes are swapped, so we swap the order as well

    # non-linearities
    def sigmoid(self, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self], 1.0,  8) #  8 = ElementWiseOperator.opSigmoid
    def tanh(self, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self], 1.0,  9) #  9 = ElementWiseOperator.opTanh
    def relu(self, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self], 1.0, 14) # 14 = ElementWiseOperator.opLinearRectifier
    def exp(self, out=None):
        return NDArrayViewOpsMixin._num_op(out, [self], 1.0, 12) # 12 = ElementWiseOperator.opExp

    # reductions
    # TODO: add copy_to_shape() or something, which is the same as reduce_sum but with a non-optional parameter
    def reduce_sum(self, reduce_to_shape=(), out=None):
        # TODO: add a test
        res = out or NDArrayView(shape=reduce_to_shape, data_type=self.dtype, device=self.device) # reduce to scalar
        res.numeric_operation_in_place(0.0, [self], 1.0, 2, 24) # 2 = ElementWiseOperator.opCopy, 28 = ElementWiseOperator.opSum
        return res
    def reduce_log_sum(self, reduce_to_shape=(), out=None):
        res = out or NDArrayView(shape=reduce_to_shape, data_type=self.dtype, device=self.device) # reduce to scalar
        res.numeric_operation_in_place(0.0, [self], 1.0, 2, 28) # 2 = ElementWiseOperator.opCopy, 28 = ElementWiseOperator.opLogSum
        return res

    # shapes, slicing, and splicing
    def reshape(self, new_shape): # note: this is not in-place (unlike Matrix::Reshape()); rename to reshaped()? reshaped_as()?
        if self.shape == new_shape:
            return self
        res = self.as_shape(new_shape)
        res.__class__ = self.__class__
        return res
    #def drop_axis(self, axis): # TODO: this is temporary; we should use begin_axis etc. in reshape() like CNTK V2
    #    shape = self.shape
    #    if axis < 0:
    #        axis += len(shape)
    #    assert shape[axis] == 1
    #    new_shape = shape[0:axis] + shape[axis+1:]
    #    return self.reshape(new_shape)
    def __setitem__(self, key, value):
        slice = self.__getitem__(key, keep_singles=True)
        slice.numeric_operation_in_place(0.0, [value], 1.0, 2, 24) # 2 = ElementWiseOperator.opCopy
    def __getitem__(self, key, keep_singles=False):
        # BUGBUG: must implement IndexError to allow for loops
        shape = self.shape
        #print('key', key)
        if not isinstance(key, tuple):
            key = (key,)
        #print('key', key)
        start_offsets = [0,] * len(shape)
        extents = list(shape)
        dims_to_keep = [True for d in shape] # axes that get passed a single index get dropped as an axis in the result
        i_off = 0

        for i, s in enumerate(key):
            #print(i, s)
            if s is Ellipsis:
                i_off = -len(shape) # indexing from back
                continue
            if isinstance(s, slice):
                begin = s.start or 0
                end   = s.stop if s.stop is not None else shape[i]
                #print(begin, '###', shape, '###', shape[i], '###', s.stop, '###', end)
                if s.step is not None and i[2] != 1:
                    raise ValueError('NDArrayView: slices with steps are not supported')
                if begin < 0:
                    begin += shape[i]
                if end < 0:
                    end += shape[i]
                start_offsets[i] = begin
                extents[i]       = end - begin
            else:
                start_offsets[i] = s
                extents[i]       = 1
                dims_to_keep[i] = False # a single index: drop this dimension
        #print('start', start_offsets, 'extents', extents)
        res = self.slice_view(tuple(start_offsets), tuple(extents), self.is_read_only)
        res.__class__ = self.__class__
        if not keep_singles and not all(dims_to_keep):
            shape = res.shape
            dims = ((dim,) * dims_to_keep[i] for i, dim in enumerate(shape))
            shape = sum(dims, ())
            res = res.reshape(shape)
        return res
    @staticmethod
    def splice(*args, axis=-1, out=None): # negative axis will insert a new axis
        # BUGBUG: axis=-1 conflicts with numpy's meaning, which will reference from the end. Maybe use None instead?
        axis = len(args[0].shape) - 1 - axis # swap to C++ API convention
        res = NDArrayView.splice_from(args, axis, out)
        res.__class__ = args[0].__class__
        return res

def _add_ndarrayview_ops(klass):
    for overload_name in ['__add__', '__sub__', '__mul__', 'greater',
                          '__radd__', '__rmul__',
                          '__iadd__', '__isub__',
                          '__matmul__',
                          'dot', 'dot_transpose', 'transpose_dot',
                          'sigmoid', 'tanh', 'relu', 'exp',
                          'reduce_sum', 'reduce_log_sum',
                          'reshape', '__getitem__', '__setitem__', 'splice']:
        # bring this back in once C++ splice() is functional
        if getattr(klass, overload_name, None):
            raise ValueError('class "%s" already has operator overload "%s"' %
                             (klass, overload_name))
        setattr(klass, overload_name, NDArrayViewOpsMixin.__dict__[overload_name])
