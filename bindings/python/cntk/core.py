# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import warnings
import numbers
import numpy as np
from scipy import sparse

from . import cntk_py
from .device import use_default_device, cpu, DeviceKind
from cntk.internal import typemap
from cntk.internal.sanitize import sanitize_batch,\
                                   _sparse_to_dense_network_cache,\
                                   data_type_to_dtype


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
        Create an :class:`NDArrayView` instance from a NumPy array.

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
    def from_csr(csr_array, device=None, read_only=False, borrow=False, shape=None):
        '''
        Create an :class:`NDArrayView` instance from a SciPy sparse array in CSR
        format.

        Args:
            csr_array (scipy.sparse.csr_matrix): SciPy sparse matrix in CSR
             format
            device (:class:`~cntk.device.DeviceDescriptor`): device this value
             should be put on
            read_only (bool, optional): whether the data can be modified or
             not (default False)
            borrow (bool, default False): whether nd_arrary memory can be
             borrowed internally to speed up the data creation
            shape (tuple, default None): shape of the created NDArrayView.
             If unspecified, the created NDArrayView has the same shape as the csr_matrix.
             Otherwise, an NDArrayView object of specified shape is created using csr_data.
             The total size and number of rows of csr_data must match specified NDArrayView shape.

        Returns:
            :class:`NDArrayView` instance
        '''
        if not sparse.isspmatrix_csr(csr_array):
            raise TypeError("only CSR is supported as of now. Please "
                            "convert your data using 'tocsr()'")

        if shape is None:
            shape = csr_array.shape

        if device is None:
            device = use_default_device()

        if shape[-1] != csr_array.shape[-1]:
            raise ValueError('csr_matrix row size (%d) does not match the least '
                             'significant axis dimension (%d) of the NDArrayView shape'
                             % (csr_array.shape[-1], shape[-1]))
           
        import functools
        import operator
        csr_array_size = functools.reduce(operator.mul, csr_array.shape)
        ndarrayview_size = functools.reduce(operator.mul, shape)
        if csr_array_size != ndarrayview_size:
            raise ValueError('csr_matrix total size (%d) does not match the total size '
                             '(%d) of the NDArrayView shape' % (csr_array_size, ndarrayview_size))
 
        return cntk_py.NDArrayView(shape, csr_array.data,
                                   csr_array.indptr, csr_array.indices, device,
                                   read_only, borrow)

    @staticmethod
    @typemap
    def from_data(data, device=None, read_only=False, borrow=False):
        '''
        Create an :class:`NDArrayView` instance from a NumPy or SciPy sparse
        array in CSR format.

        Args:
            data (numpy.ndarray or scipy.sparse.csr_matrix): data
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

    @property
    def shape(self):
        '''
        The shape of this instance.
        '''
        return super(NDArrayView, self).shape().dimensions()

    @typemap
    def slice_view(self, start_offset, extent, read_only=True):
        '''
        Returns a sliced view of the instance.

        Example:
            >>> # Creating an array of shape (1, 1, 2, 3)
            >>> np_array = np.asarray([[[[10, 20, 30], [40, 50, 60]]]], \
                                      dtype=np.float32)
            >>> nd = NDArrayView.from_dense(np_array)
            >>> sliced = nd.slice_view([0, 0, 0, 0], [2, 3])
            >>> np_sliced = sliced.asarray()
            >>> # Result is an array of shape (2, 3)
            >>> print(np_sliced)
            [[ 10.  20.  30.]
             [ 40.  50.  60.]]

        Args:
          start_offset (tuple or list): shape of the same rank as this Value
           instance that denotes the start of the slicing
          extent (tuple or list): shape of the right-aligned extent to keep
          read_only (bool): whether the returned slice is read only or not
        '''
        return super(NDArrayView, self).slice_view(
                list(reversed(start_offset)),
                list(reversed(extent)),
                read_only)

    @property
    @typemap
    def device(self):
        '''
        Retrieves the :class:`~cntk.device.DeviceDescriptor` instance.
        '''
        return super(NDArrayView, self).device()

    @property
    def is_sparse(self):
        '''
        Whether the data is sparse or dense
        '''
        return super(NDArrayView, self).is_sparse()

    @property
    def is_read_only(self):
        '''
        Whether the data is read-only
        '''
        return super(NDArrayView, self).is_read_only()

    @property
    def dtype(self):
        '''
        NumPy data type of the instance
        '''
        return data_type_to_dtype(self.get_data_type())


class Value(cntk_py.Value):
    '''
    Internal representation of minibatch data.

    Args:
        batch: batch input for `var`.
         It can be:

          * a pure Python structure (list of lists, ...),
          * a list of NumPy arrays or SciPy sparse CSR matrices
          * a :class:`~cntk.core.Value` object (e.g. returned by :func:`one_hot`)
        seq_starts (list of `bool`\ s or None): if None, every sequence is
         treated as a new sequence. Otherwise, it is interpreted as a list of
         Booleans that tell whether a sequence is a new sequence (`True`) or a
         continuation of the sequence in the same slot of the previous
         minibatch (`False`)
        device (:class:`~cntk.device.DeviceDescriptor`): device this value
         should be put on
    '''

    def __init__(self, batch, seq_starts=None, device=None):
        if device is None:
            device = use_default_device()

        if isinstance(batch, np.ndarray):
            ndav = NDArrayView.from_dense(batch, device)
        else:
            ndav = batch

        if seq_starts:
            super(Value, self).__init__(ndav, seq_starts)
        else:
            super(Value, self).__init__(ndav)

    def as_sequences(self, variable=None):
        '''
        Convert a Value to a sequence of NumPy arrays that have their masked
        entries removed.

        Returns:
            If variable contains more dynamic axes than the batch axis, a list
            of NumPy arrays (if dense) or a SciPy CSR array (if sparse) will be
            returned. Otherwise, the arrays will be returned directly.
        '''
        if self.is_sparse:
            if variable is None:
                raise ValueError('cannot convert sparse value to sequences '
                                 'without the corresponding variable')
            network = _sparse_to_dense_network_cache(variable.shape, True, self.device)

            warnings.warn('converting Value object to CSR format might be slow')

            # TODO: Add direct conversion, since creating an intermediate array might be slow
            dense_data = network.eval(self, device=self.device)
            return [sparse.csr_matrix(seq) for seq in dense_data]

        else:
            # Checking for mask without retrieving
            has_mask = super(Value, self).mask() is not None

            if has_mask:
                if variable is None:
                    mask = self.mask
                    return [seq[mask[idx] != cntk_py.MaskKind_Invalid]
                               for idx, seq in enumerate(self.data.asarray())]
                else:
                    value_sequences = self.unpack_variable_value(variable, True, cpu())
                    return [seq.asarray() for seq in value_sequences[0]]
            else:
                # This might be costly, but we need to return a list for
                # consistency.
                arr = self.asarray()
                if not arr.shape:
                    return [arr]
                else:
                    return list(arr)

    @staticmethod
    def _as_best_data_type(var, sample):
        convert_to_var_dtype = False

        if isinstance(sample, list):
            try:
                sample = asarray(sample, dtype=var.dtype)
            except ValueError:
                s = sample
                while isinstance(s, list) and len(s) > 0:
                    s = s[0]
                if sparse.issparse(s):
                    raise ValueError('if you provide sparse data, every '
                                        'sequence must be encoded as one '
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
            var (:class:`~cntk.variables.Variable`): variable into which
             ``data`` is passed
            data: data for `var`.
             It can be:

              * a single NumPy array denoting the full minibatch
              * a list of NumPy arrays or SciPy sparse CSR matrices
              * a single NumPy array denoting one parameter or constant
            seq_starts (list of `bool`\ s or None): if None, every sequence is
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
            ndav = NDArrayView.from_data(data, device=device, borrow=False)

            return cntk_py.Value(ndav)

        elif len(var.dynamic_axes) <= 1 and isinstance(data, list) and len(data) > 1:
            warnings.warn('you provided the minibatch data as a list, but '
                          'your corresponding input variable (uid "%s") has '
                          'only one dynamic axis (batch axis). To speed up '
                          'graph execution, please convert the data '
                          'beforehand into one NumPy array to speed up '
                          'training.' % var.uid)

        if isinstance(data, cntk_py.NDArrayView):
            return cntk_py.Value(data)

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

        if isinstance(data, sparse.csr_matrix):
            if seq_starts:
                raise ValueError('seq_starts are not supported with data specified as a single '
                                 'SciPy sparse CSR matrix. Please provide a list instead')
            else:
                data = Value._as_best_data_type(var, data)
                ndav = NDArrayView.from_data(data, device)

                return cntk_py.Value(ndav)

        if not isinstance(data, list):
            raise ValueError('batch must be a list of NumPy arrays or '
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

    ONE_HOT_SKIP = cntk_py.Value.one_hot_skip

    @staticmethod
    @typemap
    def one_hot(batch, num_classes, dtype=None, device=None):
        '''
        Converts ``batch`` into a :class:`~cntk.core.Value` object of ``dtype``
        such that the integer data in ``batch`` is interpreted as the indices
        representing one-hot vectors. Use Value.ONE_HOT_SKIP for a zero vector

        Example:
            >>> num_classes = 6
            >>> sparse_indices = [[1,C.Value.ONE_HOT_SKIP,5],[4]]
            >>> i0 = C.sequence.input(shape=num_classes, is_sparse=True)
            >>> z = C.times(i0, np.eye(num_classes))
            >>> value = C.Value.one_hot(sparse_indices, num_classes)
            >>> z.eval({i0: value})
            [array([[ 0.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.]], dtype=float32),
             array([[ 0.,  0.,  0.,  0.,  1.,  0.]], dtype=float32)]
            <BLANKLINE>
            >>> num_classes = 6
            >>> sample_shape = (2, num_classes)
            >>> sparse_indices = [[1,5,3,2],[4,1]]
            >>> i0 = C.sequence.input(shape=sample_shape, is_sparse=True)
            >>> z = C.times(i0, np.eye(num_classes))
            >>> value = C.Value.one_hot(sparse_indices, sample_shape)
            >>> z.eval({i0: value})
            [array([[[ 0.,  1.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  1.]],
                    [[ 0.,  0.,  0.,  1.,  0.,  0.],
                     [ 0.,  0.,  1.,  0.,  0.,  0.]]], dtype=float32),
             array([[[ 0.,  0.,  0.,  0.,  1.,  0.],
                     [ 0.,  1.,  0.,  0.,  0.,  0.]]], dtype=float32)]

        Args:
            batch (list of lists of integers): batch input data of indices
            sample_shape (int or tuple): number of classes or shape of each
             sample whose trailing axis is one_hot
            dtype (`np.float32`, `np.float64`, default None): data type
            device (:class:`~cntk.device.DeviceDescriptor`, default None): device
             this value should be put on

        Returns:
            ``batch`` converted into a :class:`~Value` object that can be passed to
            the forward or eval function.
        '''
        if device is None:
            device = use_default_device()

        if isinstance(num_classes, numbers.Integral):
            sample_shape = (num_classes,)
        else:
            sample_shape = num_classes

        if isinstance(batch, np.ndarray):
            batch = batch.tolist()
        elif isinstance(batch, list) and isinstance(batch[0], np.ndarray):
            batch = [b.tolist() for b in batch]

        try:
            elem = batch[0][0]
        except:
            raise ValueError('input must be a list of list of integers')

        if not isinstance(elem, numbers.Integral):
            data_type = type(elem)
            raise ValueError('supplied data to one_hot() must be of type integer'
                             ' and not "%s" since it is index data.' % data_type)

        if dtype in [np.float32, None]:
            value = cntk_py.Value.create_one_hot_float(
                sample_shape, batch, device, False)
        elif dtype == np.float64:
            value = cntk_py.Value.create_one_hot_double(
                sample_shape, batch, device, False)
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
        return super(Value, self).mask().asarray()

    @property
    @typemap
    def data(self):
        '''
        Retrieves the underlying :class:`NDArrayView` instance.
        '''
        return super(Value, self).data()

    def __len__(self):
        '''
        Number of samples in this value object.
        '''
        return self.shape[0]

    @property
    @typemap
    def device(self):
        '''
        Retrieves the :class:`~cntk.device.DeviceDescriptor` instance.
        '''
        return super(Value, self).device()

    @property
    def is_sparse(self):
        '''
        Whether the data is sparse or dense
        '''
        return super(Value, self).is_sparse()

    @property
    def is_read_only(self):
        '''
        Whether the data is read-only
        '''
        return super(Value, self).is_read_only()

    @property
    def dtype(self):
        '''
        NumPy data type of the instance
        '''
        return data_type_to_dtype(self.get_data_type())


def user_function(user_func):
    '''
    Wraps the passed Function to create a composite representing the
    composite Function graph rooted at the passed root Function.
    '''
    from . import as_composite
    return as_composite(user_func)


def asarray(value, dtype=None):
    '''
    Converts a Value object to a sequence of NumPy arrays (if dense) or CSR arrays (if sparse).
    '''
    if hasattr(value, 'asarray'):
        value = value.asarray()
    else:
        orig_type = type(value)
        value = np.asarray(value)
        if value.dtype == object:
            raise ValueError('could not convert instance of type %s to '
                             'an array'%orig_type)
    if dtype is not None:
        return value.astype(dtype)


def asvalue(variable, data_array):
    '''
    Converts a sequence of NumPy arrays or CSR arrays to a Value object.
    '''
    return sanitize_batch(variable, data_array)
