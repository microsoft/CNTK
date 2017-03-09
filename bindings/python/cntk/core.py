# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import warnings
import numpy as np
from scipy import sparse

from . import cntk_py
from .device import use_default_device, cpu, DeviceKind
from cntk.internal import typemap


def _is_c_contiguous(data):
    while isinstance(data, list):
        data = data[0]

    return data.flags.c_contiguous


class NDArrayView(cntk_py.NDArrayView):
    '''
    Creates an empty dense internal data representation of a
    :class:`~cntk.core.Value` object.
    To create an NDArrayView from a NumPy array, use :meth:`from_dense`.
    To create an NDArrayView from a sparse array, use :meth:`from_csr`.

    Args:
        shape (tuple): shape of the data
        data_type (np.float32, np.float64): data type of the data
        device (:class:`~cntk.device.DeviceDescriptor`): device this value
         should be put on
    '''

    def __init__(self, shape, data_type, device=None):
        from cntk.internal import sanitize_shape, sanitize_dtype_cntk
        shape = sanitize_shape(shape)
        data_type = sanitize_dtype_cntk(data_type)
        if device is None:
            device = use_default_device()
        super(NDArrayView, self).__init__(data_type, cntk_py.StorageFormat_Dense, shape,
                                          device)

    @staticmethod
    @typemap
    def from_dense(np_array, device=None, read_only=False, borrow=False):
        '''
        Create a :class:`NDArrayView` instance from a NumPy array.

        Args:
            np_array (numpy.ndarray): NumPy array
            device (:class:`~cntk.device.DeviceDescriptor`): device this value
             should be put on
            borrow (bool, default False): whether nd_arrary memory can be
             borrowed internally to speed up the data creation
            read_only (bool, optional): whether the data can be modified or
             not (default False)

        Returns:
            :class:`NDArrayView` instance
        '''
        if not isinstance(np_array, np.ndarray):
            raise TypeError('data must be of type numpy.ndarray'
                            ' and not %s' % type(np_array))

        if not _is_c_contiguous(np_array):
            warnings.warn('data is not C contiguous; rearrange your '
                          'data/computation to avoid costly data conversions',
                          RuntimeWarning)
            np_array = np.ascontiguousarray(np_array)

        if device is None:
            device = use_default_device()

        return cntk_py.NDArrayView(np_array, device, read_only, borrow)

    @staticmethod
    @typemap
    def from_csr(csr_array, device=None, read_only=False, borrow=False):
        '''
        Create a :class:`NDArrayView` instance from a SciPy sparse array in CSR
        format.

        Args:
            csr_array (scipy.sparse.csr.csr_matrix): SciPy sparse matrix in CSR
             format
            device (:class:`~cntk.device.DeviceDescriptor`): device this value
             should be put on
            read_only (bool, optional): whether the data can be modified or
             not (default False)
            borrow (bool, default False): whether nd_arrary memory can be
             borrowed internally to speed up the data creation

        Returns:
            :class:`NDArrayView` instance
        '''
        if not sparse.isspmatrix_csr(csr_array):
            raise TypeError("only CSR is supported as of now. Please "
                            "convert your data using 'tocsr()'")

        if device is None:
            device = use_default_device()

        return cntk_py.NDArrayView(csr_array.shape, csr_array.data,
                                   csr_array.indptr, csr_array.indices, device,
                                   read_only, borrow)

    @staticmethod
    @typemap
    def from_data(data, device=None, read_only=False, borrow=False):
        '''
        Create a :class:`NDArrayView` instance from a NumPy or SciPy sparse
        array in CSR format.

        Args:
            data (numpy.ndarray or scipy.sparse.csr.csr_matrix): data
            device (:class:`~cntk.device.DeviceDescriptor`): device this value
             should be put on
            read_only (bool, optional): whether the data can be modified or
             not (default False)
            borrow (bool, default False): whether nd_arrary memory can be
             borrowed internally to speed up the data creation

        Returns:
            :class:`NDArrayView` instance
        '''
        if isinstance(data, cntk_py.NDArrayView):
            return data

        if isinstance(data, np.number):
            data = np.asarray(data)

        if isinstance(data, np.ndarray):
            ndav = NDArrayView.from_dense(data, device, borrow=borrow)
        elif sparse.issparse(data):
            ndav = NDArrayView.from_csr(data, device, borrow=borrow)
        else:
            raise TypeError('data type "%s" is not supported. Please '
                            'provide the data as a Python list of NumPy '
                            'arrays or Scipy CSR matrices.' % type(data))

        return ndav


class Value(cntk_py.Value):
    '''
    Internal representation of minibatch data.

    Args:
        shape (tuple): shape of the value
        value (None or value that can be cast to NumPy array): the value to
         be converted
        dtype: data type (np.float32 or np.float64)
        batch: batch input for `var`.
         It can be:
          * a pure Python structure (list of lists, ...),
          * a list of NumPy arrays or SciPy sparse CSR matrices
          * a :class:`~cntk.core.Value` object (e.g. returned by :func:`one_hot`)
        seq_starts (list of `bool`s or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the sequence in the same slot of the previous
         minibatch (`False`)
        device (:class:`~cntk.device.DeviceDescriptor`): device this value
         should be put on
    '''

    def __init__(self, shape=None, dtype=None, batch=None, seq_starts=None, device=None):
        if device is None:
            device = use_default_device()

        if shape and dtype:
            # FIXME is this needed?
            ndav = NDArrayView(shape, dtype, device)

        elif batch:
            if isinstance(batch, np.ndarray):
                ndav = NDArrayView.from_dense(batch, device)
            else:
                ndav = batch

        if seq_starts:
            super(Value, self).__init__(ndav, seq_starts)
        else:
            super(Value, self).__init__(ndav)

    @staticmethod
    def _as_best_data_type(var, sample):
        convert_to_var_dtype = False

        if isinstance(sample, list):
            try:
                sample = np.asarray(sample, dtype=var.dtype)
            except ValueError:
                s = sample
                while isinstance(s, list) and len(s) > 0:
                    s = s[0]
                if sparse.issparse(s):
                    raise ValueError('if you provide sparse data, every '
                                     'sequence has to be encoded as one '
                                     'csr_matrix instance. Your sequence '
                                     'was: \'%s\'' % str(sample))
                else:
                    raise

            if sample.dtype != var.dtype:
                raise ValueError('could not convert sample data to '
                                 'NumPy array')

        elif sample.dtype in (np.float32, np.float64):
            if sample.dtype != var.dtype:
                convert_to_var_dtype = True

        elif np.issubdtype(sample.dtype, int):
            convert_to_var_dtype = True

        else:
            raise ValueError('only integer, float32 and float64 are '
                             'supported, you gave %s' % sample.dtype)

        if convert_to_var_dtype:
            warnings.warn('your data is of type "%s", but your input '
                          'variable (uid "%s") expects "%s". Please convert '
                          'your data beforehand to speed up training.' %
                          (sample.dtype, var.uid, str(var.dtype)))
            sample = sample.astype(var.dtype)

        return sample

    @staticmethod
    @typemap
    def create(var, data, seq_starts=None, device=None, read_only=False):
        '''
        Creates a :class:`~cntk.core.Value` object.

        Args:
            var (:class:`~cntk.ops.variables.Variable`): variable into which
             ``data`` is passed
            data: data for `var`
             It can be:
              * a single NumPy array denoting the full minibatch
              * a list of NumPy arrays or SciPy sparse CSR matrices
              * a single NumPy array denoting one parameter or constant
            seq_starts (list of `bool`s or None): if None, every sequence is
             treated as a new sequence. Otherwise, it is interpreted as a list of
             Booleans that tell whether a sequence is a new sequence (`True`) or a
             continuation of the sequence in the same slot of the previous
             minibatch (`False`)
            device (:class:`~cntk.device.DeviceDescriptor`, default None): device
             this value should be put on
            read_only (bool, default False): whether the data is read only

        Returns:
            :class:`~cntk.core.Value` object.
        '''
        if not isinstance(var, cntk_py.Variable):
            raise TypeError('Variable expected, but got "%s"' % type(var))

        if not var.dynamic_axes:
            # No dynamic axes -> we can pass everything in one go
            data = Value._as_best_data_type(var, data)
            # Since the core API's Value does not copy single NDArrayViews,
            # we cannot borrow the memory here.
            ndav = NDArrayView.from_data(data, device=cpu(), borrow=False)

            return cntk_py.Value(ndav)

        elif len(var.dynamic_axes) <= 1 and isinstance(data, list):
            warnings.warn('you provided the minibatch data as a list, but '
                          'your corresponding input variable (uid "%s") has '
                          'only one dynamic axis (batch axis). To speed up '
                          'graph executen, please convert the data '
                          'beforehand into one NumPy array to speed up '
                          ' training.' % var.uid)

        if isinstance(data, np.ndarray):
            # The outermost axis has to be Python list. If the user passes a
            # full minibatch as one NumPy array, we have to convert it.
            if data.dtype == object:
                raise ValueError('dtype object is not supported. If this is a '
                                 'batch of sequences, you need to pass them as a '
                                 'pure-Python list of NumPy arrays')

            if seq_starts:
                data = list(np.atleast_1d(data))
            else:
                data = Value._as_best_data_type(var, data)
                ndav = NDArrayView.from_data(data, device)

                return cntk_py.Value(ndav)

        if not isinstance(data, list):
            raise ValueError('batch has to be a list of NumPy arrays or '
                             'SciPy CSR matrices')

        # NDArrayViews are all created on CPU. The Value object later then will
        # move it to the requested device.
        # As Value will later create copies anyways, we do not create copies in
        # NDArrayView itself. Because of that, we need to keep around the
        # instances _as_best_data_type() until we have passed them to
        # Value_create() where it will be copied further.
        data = [Value._as_best_data_type(var, sample) for sample in data]
        borrow = device.type() == DeviceKind.CPU
        list_of_ndavs = [NDArrayView.from_data(sample, device=cpu(),
                                               borrow=borrow)
                         for sample in data]

        from cntk.internal import sanitize_shape
        value = cntk_py.Value_create(
            sanitize_shape(var.shape),
            list_of_ndavs,
            seq_starts or [],
            device or use_default_device(),
            read_only,
            True)  # always create a copy in Value

        return value

    @staticmethod
    @typemap
    def one_hot(batch, num_classes, dtype=None, device=None):
        '''
        Converts ``batch`` into a :class:`~cntk.core.Value` object of ``dtype``
        such that the integer data in ``batch`` is interpreted as the indices
        representing one-hot vectors.

        Example:
            >>> num_classes = 6
            >>> sparse_indices = [[1,5],[4]]
            >>> i0 = C.input_variable(shape=num_classes, is_sparse=True)
            >>> z = C.times(i0, np.eye(num_classes))
            >>> value = C.one_hot(sparse_indices, num_classes)
            >>> z.eval({i0: value})
            [array([[ 0.,  1.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]], dtype=float32), array([[ 0.,  0.,  0.,  0.,  1.,  0.]], dtype=float32)]

        Args:
            batch (list of lists of integers): batch input data of indices
            num_classes (int): number of classes
            dtype (`np.float32`, `np.float64`, default None): data type
            device (:class:`~cntk.device.DeviceDescriptor`, default None): device
             this value should be put on

        Returns:
            ``batch`` converted into a :class:`~Value` object that can be passed to
            the forward or eval function.
        '''
        if device is None:
            device = use_default_device()

        if isinstance(batch, np.ndarray):
            batch = batch.tolist()

        try:
            data_type = type(batch[0][0])
        except:
            raise ValueError('input must be a list of list of integers')

        if data_type != int:
            raise ValueError('supplied data to one_hot() must be of type integer'
                             ' and not "%s" since it is index data.' % data_type)

        if dtype in [np.float32, None]:
            value = cntk_py.Value.create_one_hot_float(
                num_classes, batch, device, False)
        elif dtype == np.float64:
            value = cntk_py.Value.create_one_hot_double(
                num_classes, batch, device, False)
        return value

    @property
    def shape(self):
        '''
        The rectangular shape of this value. I.e., if this value has sequences
        of varying lengths, the shape will have the max sequence length in the
        sequence dimension.
        '''
        return super(Value, self).shape().dimensions()

    @property
    def mask(self):
        '''
        The mask matrix of this value. Each row denotes a sequence with its
        elements describing the mask of the element:
         * 2: beginning of sequence (e.g. an LSTM would be reset)
         * 1: valid element
         * 0: invalid element

        Example:
          A mask of ``[[2, 1, 1], [1, 1, 0]]`` describes a batch of two
          sequences. The first has three elements, of which the first element
          (2) signals the beginning of a sequence. The second sequence has two
          elements (last element marked 'invalid' by '0'). As it starts with
          (1), it is a continuation of the 2nd sequence in the previous
          minibatch.
        '''
        return np.asarray(super(Value, self).mask())

    def __len__(self):
        '''
        Number of samples in this value object.
        '''
        return self.shape[0]


def user_function(user_func):
    '''
    Wraps the passed Function to create a composite representing the
    composite Function graph rooted at the passed root Function.
    '''
    from . import as_composite
    return as_composite(user_func)
