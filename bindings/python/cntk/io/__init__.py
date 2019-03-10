# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
CNTK IO utilities.
"""


import warnings
from cntk import cntk_py, Value
from cntk.tensor import ArrayMixin
from cntk.internal import typemap, sanitize_dtype_cntk, is_string
from cntk.device import use_default_device
from cntk.logging import TraceLevel, get_trace_level
from cntk.variables import Record
from cntk.internal.utils import _py_dict_to_cntk_dict
import cntk.io.transforms

import numpy as np
import uuid

INFINITELY_REPEAT = cntk_py.MinibatchSource.infinitely_repeat
'''int: constant used to specify a minibatch scheduling unit to equal the size of the full data sweep.'''

FULL_DATA_SWEEP = cntk_py.MinibatchSource.full_data_sweep
DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS = cntk_py.MinibatchSource.default_randomization_window_in_chunks


class MinibatchData(cntk_py.MinibatchData, ArrayMixin):

    '''
    Holds a minibatch of input data. This is never directly created, but
    only returned by :class:`MinibatchSource` instances.
    '''

    def __init__(self, value, num_sequences, num_samples, sweep_end):
        super(MinibatchData, self).__init__(value, num_sequences, num_samples,
                                            sweep_end)

    @property
    def num_sequences(self):
        '''
        The number of sequences in this minibatch
        '''
        return self.number_of_sequences

    @property
    def num_samples(self):
        '''
        The number of samples in this minibatch
        '''
        return self.number_of_samples

    def as_sequences(self, variable=None):
        '''
        Convert the value of this minibatch instance to a sequence of NumPy
        arrays that have their masked entries removed.

        Returns:
            a list of NumPy arrays if dense, otherwise a SciPy CSR array
        '''
        return self.data.as_sequences(variable)

    @property
    def shape(self):
        '''
        The shape of the data in this minibatch as tuple.
        '''
        return self.data.shape

    @property
    @typemap
    def data(self):
        '''
        Retrieves the underlying :class:`~cntk.core.Value` instance.
        '''
        return super(MinibatchData, self).data

    @property
    def mask(self):
        '''
        The mask object of the minibatch. In it, `2` marks the beginning of a
        sequence, `1` marks a sequence element as valid, and `0` marks it as
        invalid.
        '''
        return self.data.mask

    @property
    def end_of_sweep(self):
        '''
        Indicates whether the data in this minibatch comes from a sweep end
        or crosses a sweep boundary (and as a result includes data from
        different sweeps).
        '''
        return self.sweep_end

    @property
    def is_sparse(self):
        '''
        Whether the data in this minibatch is sparse.
        '''
        return self.data.is_sparse

    def __len__(self):
        return self.num_sequences


class MinibatchSource(cntk_py.MinibatchSource):
    '''
    MinibatchSource(deserializers, max_samples=cntk.io.INFINITELY_REPEAT, max_sweeps=cntk.io.INFINITELY_REPEAT, randomization_window_in_chunks=cntk.io.DEFAULT_RANDOMIZATION_WINDOW, randomization_window_in_samples=0, randomization_seed=0, trace_level=cntk.logging.get_trace_level(), multithreaded_deserializer=None, frame_mode=False, truncation_length=0, randomize=True, max_errors=0)

    Args:
        deserializers (a single deserializer or a `list`): deserializers to be used in the composite reader
        max_samples (`int`, defaults to :const:`cntk.io.INFINITELY_REPEAT`): The maximum number of input samples
          (not 'label samples') the reader can produce. After this number has been reached, the reader
          returns empty minibatches on subsequent calls to :meth:`next_minibatch`. `max_samples` and `max_sweeps`
          are mutually exclusive, an exception will be raised if both have non-default values.
          **Important:**
          Click :cntkwiki:`here <BrainScript-epochSize-and-Python-epoch_size-in-CNTK>`
          for a description of input and label samples.
        max_sweeps (`int`, defaults to :const:`cntk.io.INFINITELY_REPEAT`): The maximum number of sweeps over
          the input dataset After this number has been reached, the reader returns empty minibatches on
          subsequent calls to func:`next_minibatch`. `max_samples` and `max_sweeps` are mutually exclusive,
          an exception will be raised if both have non-default values.
        randomization_window_in_chunks (`int`, defaults to :const:`cntk.io.DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS`):
          size of the randomization window in chunks, non-zero value enables randomization.
          `randomization_window_in_chunks` and `randomization_window_in_samples` are mutually exclusive,
          an exception will be raised if both have non-zero values.
        randomization_window_in_samples (`int`, defaults to `0`): size of the randomization window in samples,
          non-zero value enables randomization.
          `randomization_window_in_chunks` and `randomization_window_in_samples` are mutually exclusive,
          an exception will be raised if both have non-zero values.
        randomization_seed (`int`, defaults to 0): initial randomization seed value (incremented every sweep when
            the input data is re-randomized).
        trace_level (an instance of :class:`cntk.logging.TraceLevel`): the output verbosity level, defaults to
          the current logging verbosity level given by :func:`~cntk.logging.get_trace_level`.
        multithreaded_deserializer (`bool`): specifies if the deserialization should be
          done on a single or multiple threads. Defaults to `None`, which is effectively "auto" (multhithreading 
          is disabled unless ImageDeserializer is present in the deserializers list). `False` and `True` 
          faithfully turn the multithreading off/on.
        frame_mode (`bool`, defaults to `False`): switches the frame mode on and off. If the frame mode
          is enabled the input data will be processed as individual frames ignoring all sequence information
          (this option cannot be used for BPTT, an exception will be raised if frame mode is enabled and the
          truncation length is non-zero).
        truncation_length (`int`, defaults to `0`): truncation length in samples, non-zero value enables
          the truncation (only applicable for BPTT, cannot be used in frame mode, an exception will be raised
          if frame mode is enabled and the truncation length is non-zero).
        randomize (`bool`, defaults to `True`): Enables or disables randomization; use randomization_window_in_chunks or
          randomization_window_in_samples to specify the randomization range
        max_errors (`int`, defaults to `0`): maximum number of errors in the dataset to ignore
    '''
    _runtime_deserializer_table = {}
    _deserializer_factory = None
    _deserializer_counter = 0

    @staticmethod
    def _create_deserializer(id):
        # Return previosly registred object to C++ side.
        deserializer = MinibatchSource._runtime_deserializer_table[id]
        del MinibatchSource._runtime_deserializer_table[id]
        return deserializer

    @staticmethod
    def _get_config(deserializer):
        # Create and register deserializer factory if does not exists.
        from cntk.internal import _DeserializerFactory
        if MinibatchSource._deserializer_factory is None:
            MinibatchSource._deserializer_factory = _DeserializerFactory(MinibatchSource._create_deserializer)
            cntk_py._register_deserializer_factory(MinibatchSource._deserializer_factory)

        # Remember deserializer with a unique generated id
        # to return it later when _create_deserializer is called from C++ side.
        id = str(MinibatchSource._deserializer_counter)
        MinibatchSource._deserializer_counter += 1
        MinibatchSource._runtime_deserializer_table[id] = deserializer
        # Currently UserDeserializer in python does not support composability
        import _cntk_py
        d = { 'type' : id, 'module': _cntk_py.__file__, 'composable': 'false' }
        return cntk.utils._py_dict_to_cntk_dict(d)

    def __init__(self,
        deserializers,
        max_samples = INFINITELY_REPEAT,
        max_sweeps = INFINITELY_REPEAT,
        randomization_window_in_chunks = DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
        randomization_window_in_samples = 0,
        randomization_seed=0,
        trace_level = TraceLevel.Warning,
        multithreaded_deserializer=None,
        frame_mode=False,
        truncation_length=0,
        randomize=True,
        max_errors=0):

        if not isinstance(deserializers, (list,tuple)):
            deserializers = [ deserializers ]

        user_deserializers = [d for d in deserializers if isinstance(d, UserDeserializer)]
        deserializers = [d if not isinstance(d, UserDeserializer) else MinibatchSource._get_config(d) for d in deserializers]
        if len(user_deserializers) >= 1 and len(deserializers) != 1:
            raise ValueError('Currently composition for user defined deserializers is not supported.')            

        config = cntk_py.MinibatchSourceConfig(deserializers)
        config.max_samples = max_samples
        config.max_sweeps = max_sweeps
        config.randomization_window_in_chunks = randomization_window_in_chunks
        config.randomization_window_in_samples = randomization_window_in_samples
        config.randomization_seed = randomization_seed;

        if multithreaded_deserializer is not None:
            config.is_multithreaded.set(multithreaded_deserializer)

        config.is_frame_mode_enabled = frame_mode
        config.truncation_length = truncation_length

        if isinstance(trace_level, TraceLevel):
            trace_level = trace_level.value

        config.trace_level = trace_level
        config.max_errors = max_errors

        if not randomize:
            config.randomization_window_in_chunks = 0
            config.randomization_window_in_samples = 0

        source = cntk_py.create_composite_minibatch_source(config)
        # transplant into this class instance
        self.__dict__ = source.__dict__
        self._streams = None
        self._last_mb_data = None

    def stream_infos(self):
        '''
        Describes the streams this minibatch source produces.

        Returns:
            A list of instances of :class:`StreamInformation`
        '''
        return super(MinibatchSource, self).stream_infos()

    @property
    def streams(self):
        '''
        Describes the streams 'this' minibatch source produces.

        Returns:
            A `dict` mapping input names to instances of
            :class:`StreamInformation`
        '''
        if self._streams is None:
            self._streams = Record(**dict((info.m_name, info) for info in  self.stream_infos()))

        return self._streams

    def stream_info(self, name):
        '''
        Gets the description of the stream with given name.
        Throws an exception if there are none or multiple streams with this
        same name.

        Args:
            name (str): stream name to fetch

        Returns:
            :class:`StreamInformation`
            The information for the given stream name.
        '''
        return super(MinibatchSource, self).stream_info(name)

    def __getitem__(self, name):
        '''
        Return the :class:`StreamInformation` for the given
        stream name.

        Args:
            name (str): stream name to fetch
              :class:`StreamInformation` for
        '''
        return self.stream_info(name)

    @typemap
    def next_minibatch(self, minibatch_size_in_samples,
                       input_map=None, device=None, num_data_partitions=None,
                       partition_index=None):
        '''
        Reads a minibatch that contains data for all input streams.  The
        minibatch size is specified in terms of #samples and/or #sequences for
        the primary input stream; value of 0 for #samples/#sequences means
        unspecified.  In case the size is specified in terms of both #sequences
        and #samples, the smaller of the 2 is taken.  An empty map is returned
        when the MinibatchSource has no more data to return.

        Args:
            minibatch_size_in_samples (int): number of samples to retrieve for
              the next minibatch. Must be > 0.
              **Important:**
              Click :cntkwiki:`here <BrainScript-minibatchSize-and-Python-minibatch_size_in_samples-in-CNTK>` for a full description of this parameter.
            input_map (dict): mapping of :class:`~cntk.variables.Variable`
              to :class:`StreamInformation` which will be used to convert the
              returned data.
            device (`DeviceDescriptor`, defaults to `None`): CNTK DeviceDescriptor
            num_data_partitions: Used for distributed training, indicates into how many partitions
              the source should split the data.
            partition_index (`int`, defaults to `None`): Used for distributed training, indicates data from which partition to take.

        Returns:
            cntk.io.MinibatchData:
             A mapping of :class:`StreamInformation` to :class:`MinibatchData` if
             `input_map` was not specified. Otherwise, the returned value will
             be a mapping of :class:`~cntk.variables.Variable` to class:`MinibatchData`.
             When the maximum number of epochs/samples is exhausted, the return value is an empty dict.
        '''
        if self._last_mb_data is not None:
            self._last_mb_data.clear()

        if device is None:
            device = use_default_device()

        if num_data_partitions is None:
            num_data_partitions = 1

        if partition_index is None:
            partition_index = 0

        parent_inst = super(MinibatchSource, self)
        mb = parent_inst.get_next_minibatch(0,
                                            minibatch_size_in_samples,
                                            num_data_partitions,
                                            partition_index, device)

        if not mb:
            return mb

        if not input_map:
            return mb

        # We copy minibatch data here,
        # we need to make sure it is cleaned when next_minibatch
        # is called next time.       
        self._last_mb_data = {key: mb[value] for (key, value) in input_map.items()}
        return self._last_mb_data

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            A dict that has the checkpoint state of the MinibatchSource
        '''
        return super(MinibatchSource, self).get_checkpoint_state()

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (dict): checkpoint to restore from
        '''
        super(MinibatchSource, self).restore_from_checkpoint(_py_dict_to_cntk_dict(checkpoint))

    @property
    def current_position(self):
        '''
        Gets current position in the minibatch source.

        Args:
            getter (:class:`~cntk.cntk_py.Dictionary`): minibatch position on the
             global timeline.
            setter (:class:`~cntk.cntk_py.Dictionary`): position returned by
             the getter
        '''
        return self.get_checkpoint_state()

    @current_position.setter
    def current_position(self, position):
        self.restore_from_checkpoint(position)


class StreamInformation(cntk_py.StreamInformation):
    '''
    Stream information container that is used to describe streams when
    implementing custom minibatch source through :class:`UserMinibatchSource`.

    Args:
        name (str): name of the stream
        stream_id (int): unique ID of the stream
        storage_format (str): 'dense' or 'sparse'
        dtype (NumPy type): data type
        shape (tuple): shape of the elements
        defines_mb_size (bool, default to False): whether this stream defines the minibatch size when there are multiple
          streams.
    '''

    _storage = {'dense': cntk_py.StorageFormat_Dense,
                'sparse': cntk_py.StorageFormat_SparseCSC}

    def __init__(self, name, stream_id, storage_format, dtype,
                 shape, defines_mb_size=False):
        super(StreamInformation, self).__init__()
        self.m_name = name
        self.m_id = stream_id
        self.m_storage_format = StreamInformation._storage[storage_format]
        self.m_element_type = sanitize_dtype_cntk(dtype)
        # raw NDShape is column based, so we need to reverse dimensions.
        self.m_sample_layout = cntk_py.NDShape(list(reversed(shape)))
        self.sample_shape = shape
        self.storage_format = storage_format
        self.m_defines_mb_size = defines_mb_size

    @property
    def name(self):
        return self.m_name

class UserMinibatchSource(cntk_py.SwigMinibatchSource):
    '''
    Base class of all user minibatch sources.
    '''
    def __init__(self):
        super(UserMinibatchSource, self).__init__()

        streams = {si.m_name: si for si in self.stream_infos()}
        self.streams = Record(**streams)

    def stream_infos(self):
        '''
        Function to be implemented by the user.

        Returns:
            list of :class:`StreamInformation` instances
        '''
        raise NotImplementedError

    def _stream_infos(self, sinfos=None):
        # sinfos is a list of stream information, which we need to fill in
        # place, # because Swig demands it that way.
        sinfos.extend(self.stream_infos())

    def stream_info(self, name):
        '''
        Gets the description of the stream with given name.
        Throws an exception if there are none or multiple streams with this
        same name.
        '''
        return super(UserMinibatchSource, self).stream_info(name)

    def next_minibatch(self, num_samples, number_of_workers, worker_rank, device=None):
        '''
        Function to be implemented by the user.

        Args:
            num_samples (int): number of samples to return
            number_of_workers (int): number of workers in total
            worker_rank (int): worker for which the data is to be returned
            device (`DeviceDescriptor`, defaults to `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is performed. If `None`, the default device is used.

        Returns:
            mapping of :class:`StreamInformation` to :class:`MinibatchData`
        '''
        raise NotImplementedError

    def _next_minibatch(self, info_map, mb_size_in_sequences,
            mb_size_in_samples, number_of_workers, worker_rank, device):
        # mbsize_in_sequences is ignored

        mb = self.next_minibatch(mb_size_in_samples, number_of_workers, worker_rank, device)
        info_map.update(mb)

    def _get_checkpoint_state(self):
        state = self.get_checkpoint_state()
        d = cntk_py.Dictionary()
        for key, val in state.items():
            if not is_string(key):
                raise ValueError('the keys of the checkpoint dictionary must '
                                 'be strings. You gave "%s" of type %s' %
                                 (key, type(key)))
            dv = cntk_py.DictionaryValue(val)
            d.add(key, dv)

        return d

    def get_checkpoint_state(self):
        '''
        Returns a dictionary describing the current state of the minibatch
        source. Needs to be overwritten if the state of the minibatch source
        needs to be stored to and later restored from the checkpoint.

        Returns:
            dictionary, that can be later used on :meth:`restore_from_checkpoint`.
        '''
        return {}

    def restore_from_checkpoint(self, state):
        '''
        Sets the state of the checkpoint.

        Args:
            state (dict): dictionary containing the state
        '''
        if state:
            raise NotImplementedError('in order to use checkpointing on '
                'UserMinibatchSource, you need to implement '
                'restore_from_checkpoint(checkpoint)')

    def __getitem__(self, name):
        '''
        Return the :class:`StreamInformation` for the given
        stream name.

        Args:
            name (str): stream name to fetch
              :class:`StreamInformation` for
        '''
        return self.stream_info(name)

    def is_infinite(self):
        '''
        Should return true if the user has not specified any limit on the number of sweeps and samples.
        '''
        return False


class MinibatchSourceFromData(UserMinibatchSource):
    '''
    This wraps in-memory data as a CNTK MinibatchSource object (aka "reader"), used to feed the data into a TrainingSession.

    Use this if your data is small enough to be loaded into RAM in its entirety, and
    the data is already sufficiently randomized.

    While CNTK allows user code to iterate through minibatches by itself and feed data minibatch
    by minibatch through :func:`~cntk.train.trainer.Trainer.train_minibatch`, the standard way is to iterate
    through data using a MinibatchSource object. For example, the high-level :class:`~cntk.train.training_session.TrainingSession`
    interface, which manages a full training including checkpointing and cross validation, operates on this level.

    A MinibatchSource created as a `MinibatchSourceFromData` linearly iterates through the data provided by
    the caller as numpy arrays or scipy.sparse.csr_matrix objects, without randomization.
    The data is not copied, so if you want to modify the data while being read through a `MinibatchSourceFromData`,
    please pass a copy.

    Example:
     >>> N = 5
     >>> X = np.arange(3*N).reshape(N,3).astype(np.float32) # 6 rows of 3 values
     >>> s = C.io.MinibatchSourceFromData(dict(x=X), max_samples=len(X))
     >>> mb = s.next_minibatch(3) # get a minibatch of 3
     >>> d = mb[s.streams['x']]
     >>> d.data.asarray()
     array([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.]], dtype=float32)
     >>> mb = s.next_minibatch(3) # note: only 2 left
     >>> d = mb[s.streams['x']]
     >>> d.data.asarray()
     array([[  9.,  10.,  11.],
            [ 12.,  13.,  14.]], dtype=float32)
     >>> mb = s.next_minibatch(3)
     >>> mb
     {}

     >>> # example of a sparse input
     >>> Y = np.array([i % 3 == 0 for i in range(N)], np.float32)
     >>> import scipy.sparse
     >>> Y = scipy.sparse.csr_matrix((np.ones(N,np.float32), (range(N), Y)), shape=(N, 2))
     >>> s = C.io.MinibatchSourceFromData(dict(x=X, y=Y)) # also not setting max_samples -> will repeat
     >>> mb = s.next_minibatch(3)
     >>> d = mb[s.streams['y']]
     >>> d.data.asarray().todense()
     matrix([[ 0.,  1.],
             [ 1.,  0.],
             [ 1.,  0.]], dtype=float32)
     >>> mb = s.next_minibatch(3) # at end only 2 sequences
     >>> d = mb[s.streams['y']]
     >>> d.data.asarray().todense()
     matrix([[ 0.,  1.],
             [ 1.,  0.]], dtype=float32)

     >>> # if we do not set max_samples, then it will start over once the end is hit
     >>> mb = s.next_minibatch(3)
     >>> d = mb[s.streams['y']]
     >>> d.data.asarray().todense()
     matrix([[ 0.,  1.],
             [ 1.,  0.],
             [ 1.,  0.]], dtype=float32)

     >>> # values can also be GPU-side CNTK Value objects (if everything fits into the GPU at once)
     >>> s = C.io.MinibatchSourceFromData(dict(x=C.Value(X), y=C.Value(Y)))
     >>> mb = s.next_minibatch(3)
     >>> d = mb[s.streams['y']]
     >>> d.data.asarray().todense()
     matrix([[ 0.,  1.],
             [ 1.,  0.],
             [ 1.,  0.]], dtype=float32)

     >>> # data can be sequences
     >>> import cntk.layers.typing
     >>> XX = [np.array([1,3,2], np.float32),np.array([4,1], np.float32)]  # 2 sequences
     >>> YY = [scipy.sparse.csr_matrix(np.array([[0,1],[1,0],[1,0]], np.float32)), scipy.sparse.csr_matrix(np.array([[1,0],[1,0]], np.float32))]
     >>> s = cntk.io.MinibatchSourceFromData(dict(xx=(XX, cntk.layers.typing.Sequence[cntk.layers.typing.tensor]), yy=(YY, cntk.layers.typing.Sequence[cntk.layers.typing.tensor])))
     >>> mb = s.next_minibatch(3)
     >>> mb[s.streams['xx']].data.asarray()
     array([[ 1.,  3.,  2.]], dtype=float32)
     >>> mb[s.streams['yy']].data.shape # getting sequences out is messy, so we only show the shape
     (1, 3, 2)

    Args:
        data_streams: name-value pairs
        max_samples (`int`, defaults to :const:`cntk.io.INFINITELY_REPEAT`): The maximum number of samples
          the reader can produce. If inputs are sequences, and the different streams have different
          lengths, then each sequence counts with the maximum length.
          After this number has been reached, the reader
          returns empty minibatches on subsequent calls to :meth:`next_minibatch`.
          **Important:**
          Click :cntkwiki:`here <BrainScript-epochSize-and-Python-epoch_size-in-CNTK>`
          for a description of input and label samples.

    Returns:
     An implementation of a :class:`cntk.io.MinibatchSource` that will iterate through the data.
    '''
    def __init__(self, data_streams, max_samples = INFINITELY_REPEAT):
        from cntk import Variable
        if not data_streams:
            raise(ValueError('at least one stream must be specified, in the form name=data or name=(data, type)'))
        self._data = dict()         # [name] -> numpy.array or scipy.sparse.csr_matrix
        self._types = dict()        # [name] -> Variable._Type
        self._is_sequence = dict()  # [name] -> bool
        self._vars = dict()         # [name] -> Variable
        self._max_samples = max_samples

        # get the data and types from the input, and form streams array
        self._num_samples = -1  # total number of samples --must be the same for all args
        from scipy import sparse
        for name, arg in data_streams.items():
            if isinstance(arg, tuple):
                value, type = arg
                type = Variable._Type._sanitize(type)
                dynamic_axes = getattr(type, 'dynamic_axes', None)
                is_sequence = dynamic_axes and len(dynamic_axes) > 1
                if not isinstance(type, Variable._Type):
                    raise ValueError('type must be a CNTK variable type, e.g. Tensor[13]')
            else:
                value = arg
                is_sequence = False  # data without type cannot have a dynamic axis
                type = Variable._Type(is_sparse=isinstance(value, sparse.csr_matrix)) # shape implanted below
            if not isinstance(value[0] if isinstance(value, list) else value, (np.ndarray, sparse.csr_matrix, Value)):
                raise TypeError('data must be a numpy.array or scipy.sparse.csr_matrix, or a list of those')
            sample_shape = value[0].shape[1:] if is_sequence else value.shape[1:]
            if not type.shape_is_known:
                type = type.updated_with(shape=sample_shape) # implant the shape
            elif type.shape != sample_shape:
                ValueError("specified type's shape does not match the data's shape")
            try:
                dtype = value.dtype # numpy array and Value
            except:
                dtype = value[0].dtype # for lists
            try:
                type.dtype
            except:
                type = type.updated_with(dtype=dtype) # implant the dtype
            num_samples = MinibatchSourceFromData._get_len(value)
            if self._num_samples == -1:
                if num_samples == 0:
                    raise(ValueError('data is empty'))
                self._num_samples = num_samples
            elif self._num_samples != num_samples:
                raise TypeError('all data items must have the same first dimension')
            self._data[name] = value
            self._types[name] = type
            self._is_sequence[name] = is_sequence

        self._cursor = 0            # current position
        self._total_num_samples = 0 # total count; once the limit is reached, we stop returning data

        super(MinibatchSourceFromData, self).__init__()

    @staticmethod
    def _get_len(value): # helper to determine the length of the corpus
        try:
            return len(value) # if input is list
        except:
            return value.shape[0] # if input is csr_matrix

    def stream_infos(self):
        return [StreamInformation(name, i, ['dense', 'sparse'][getattr(self._types[name], 'is_sparse', False)], 
                                  self._types[name].dtype, self._types[name].shape)
                for i, name in enumerate(self._data.keys())]

    def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):        
        if self._total_num_samples >= self._max_samples:
            return {}
        # determine how many samples, starting from self._cursor, will fit into the requested minibatch size of num_samples
        begin = self._cursor
        end = self._cursor
        assert begin < self._num_samples
        actual_num_samples = { name: 0 for name in self._data.keys() }
        while end < self._num_samples: 
            new_num_samples = { name: actual_num_samples[name] + (MinibatchSourceFromData._get_len(value[end]) if self._is_sequence[name] else 1)
                                for name, value in self._data.items() }
            # return up to requested number of samples. but at least one even if longer
            # also stop if we hit the maximum requested number of samples
            max_num_samples = max(new_num_samples.values())
            if actual_num_samples and (max_num_samples > num_samples or self._total_num_samples + max_num_samples > self._max_samples):
                break
            actual_num_samples = new_num_samples
            end += 1

        self._total_num_samples += max(actual_num_samples.values())

        # the minibatch data to return
        result = {}  # [stream_info] -> MinibatchData
        at_end = (end == self._num_samples)
        for si in self.streams.values():
            arg = self._data[si.name]
            if isinstance(arg, Value):  # if entire corpus is one big Value, then slice NDArrayView directly
                data = arg.data
                sub_shape = data.shape[1:]
                extent = (end - begin,) + sub_shape
                start_offset = (begin,) + tuple(0 for _ in sub_shape)
                if number_of_workers != 1: # slice_view presently does not support strides
                    raise ValueError('distributed reading from Value objects is not supported')
                mb_data = data.slice_view(start_offset, extent, data.is_read_only)
            else:
                # in case of distributed reading, we sub-slice the minibatch
                #print('rank/worker', worker_rank, number_of_workers, 'reading', slice(begin+worker_rank, end+worker_rank, number_of_workers))
                mb_data = arg[begin+worker_rank:end+worker_rank:number_of_workers]
                if number_of_workers != 1:
                    mb_data = mb_data.copy() # un-stride it, to avoid performance warning
            if isinstance(mb_data, list): # create a Value object
                if si.name not in self._vars: # this case is more complex, we need a CNTK Variable
                    from cntk import input_variable, device
                    self._vars[si.name] = input_variable(**self._types[si.name])
                value = Value.create(self._vars[si.name], mb_data)
            else:
                value = Value(mb_data)
            result[si] = MinibatchData(value, num_sequences=end - begin, num_samples=actual_num_samples[si.name],
                                       sweep_end=at_end or (self._total_num_samples >= self._max_samples))

        # wrap around the cursor
        self._cursor = 0 if at_end else end

        return result

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            cntk.cntk_py.Dictionary:
            A :class:`~cntk.cntk_py.Dictionary` that has the checkpoint state
            of the MinibatchSource
        '''
        return dict(cursor=self._cursor, total_num_samples=self._total_num_samples)

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (:class:`~cntk.cntk_py.Dictionary`): checkpoint to restore from
        '''
        self._cursor = checkpoint['cursor']
        self._total_num_samples = checkpoint['total_num_samples']


def HTKFeatureDeserializer(streams):
    '''
    Configures the HTK feature reader that reads speech data from scp files.

    Args:
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          a feature stream.
    '''
    feat = []
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None:
            raise ValueError("HTKFeatureDeserializer does not support stream names")
        if 'scp' not in stream:
            raise ValueError("No scp files specified for HTKFeatureDeserializer")
        dimension = stream.dim
        scp_file = stream['scp']
        broadcast = stream['broadcast'] if 'broadcast' in stream else False
        defines_mb_size = stream.get('defines_mb_size', False)
        max_sequence_length = stream.get('max_sequence_length', 65535)
        left_context, right_context = stream.context if 'context' in stream\
                                                     else (0, 0)
        htk_config = cntk_py.HTKFeatureConfiguration(stream_name, scp_file,
                                                     dimension, left_context,
                                                     right_context, broadcast,
                                                     defines_mb_size, max_sequence_length)
        feat.append(htk_config)

    if len(feat) == 0:
        raise ValueError("no feature streams found")
    return cntk_py.htk_feature_deserializer(feat)


def HTKMLFDeserializer(label_mapping_file, streams, phoneBoundaries = False):
    '''
    Configures an HTK label reader that reads speech HTK format MLF (Master
    Label File)

    Args:
        label_mapping_file (str): path to the label mapping file
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          a label stream.
        phoneBoundaries (`bool`, defaults to False): if phone boundaries should be considered (should be set to True for CTC training, False otherwise)
    '''
    if len(streams) != 1:
        raise ValueError("HTKMLFDeserializer only accepts a single stream")
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None:
            raise ValueError("HTKMLFDeserializer does not support stream names")
        dimension = stream.dim
        if 'mlf' not in stream:
            raise ValueError(
                "No master label files specified for HTKMLFDeserializer")
        master_label_files = stream['mlf']
        if not isinstance(master_label_files, list):
            master_label_files = [master_label_files]
        return cntk_py.htk_mlf_deserializer(stream_name, label_mapping_file, dimension, master_label_files, phoneBoundaries)

def HTKMLFBinaryDeserializer(streams):
    '''
    Configures a binary HTK label reader that reads speech MLF (Master
    Label File)

    Args:
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          a label stream.
    '''
    if len(streams) != 1:
        raise ValueError("HTKMLFBinaryDeserializer only accepts a single stream")
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None:
            raise ValueError("HTKMLFDeserializer does not support stream names")
        dimension = stream.dim
        if 'mlf' not in stream:
            raise ValueError(
                "No master label files specified for HTKMLFDeserializer")
        master_label_files = stream['mlf']
        if not isinstance(master_label_files, list):
            master_label_files = [master_label_files]
        return cntk_py.htk_mlf_binary_deserializer(stream_name, master_label_files, dimension)

def LatticeDeserializer(lattice_index_file, streams):
    '''
    Configures a lattice deserializer

    Args:
        lattice_index_file (str): path to the file containing list of lattice TOC (table of content) files
    '''
    if len(streams) != 1:
        raise ValueError("LatticeDeserializer only accepts a single stream")
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None:
            raise ValueError("LatticeDeserializer does not support stream alias")
        return cntk_py.lattice_deserializer(stream_name, lattice_index_file)

def _process_image_deserializer_args(filename, streams, deserializer):
    image_stream_name = None

    # Streams with the same name are not allowed, make sure the default is
    # unique.
    label_stream_name = '_ignore_labels_' + str(uuid.uuid1())
    num_labels = 2
    transforms = []
    for key in streams:
        s = streams[key]
        alias = s.stream_alias
        if alias == "image":
            image_stream_name = key
            transforms = s.transforms
        elif alias == "label":
            label_stream_name = key
            num_labels = s.dim
        else:
            raise ValueError(
                "{}: invalid field name '{}', allowed are "
                "'image' and 'label'".format(deserializer, alias))

    if image_stream_name is None:
        raise ValueError("{}: stream name ('image' or 'label') must be "
            "specified".format(deserializer))

    return (filename, label_stream_name, num_labels,
        image_stream_name, transforms)

def ImageDeserializer(filename, streams):
    '''
    Configures the image reader that reads images and corresponding
    labels from a file of the form::

         <full path to image> <tab> <numerical label (0-based class id)>

    or::

        sequenceId <tab> path <tab> label

    Args:
        filename (str): file name of the map file that associates images to
         classes

    See also:
        :cntkwiki:`Image reader definition <BrainScript-Image-reader>`
    '''
    args = _process_image_deserializer_args(filename, streams,
        'ImageDeserializer')
    return cntk_py.image_deserializer(*args)

def Base64ImageDeserializer(filename, streams):
    '''
    Configures the image reader that reads base64 encoded images and corresponding
    labels from a file of the form::

        [sequenceId <tab>] <numerical label (0-based class id)> <tab> <base64 encoded image>

    Similarly to the ImageDeserializer, the sequenceId prefix is optional and can be omitted.

    Args:
        filename (str): file name of the input file dataset that contains images 
         and corresponding labels

    See also:
        :cntkwiki:`Base64ImageDeserializer options <BrainScript-and-Python---Understanding-and-Extending-Readers#base64imagedeserializer-options>`
    '''
    args = _process_image_deserializer_args(filename, streams,
        'Base64ImageDeserializer')
    return cntk_py.base64_image_deserializer(*args)

def CTFDeserializer(filename, streams):
    '''
    Configures the CNTK text-format reader that reads text-based files with
    lines of the form::

        [Sequence_Id] (Sample)+

    where::

        Sample=|Input_Name (Value )*

    Args:
        filename (str): file name containing the text input
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          an input stream.

    See also:
        :cntkwiki:`CNTKTextReader format <BrainScript-CNTKTextFormat-Reader>`
    '''
    for k, s in streams.items():
        if s.stream_alias is None:
            raise ValueError("CTFDeserializer: stream name for key %s must be "
                             "specified" % k)
    sc = [cntk_py.StreamConfiguration(
        k, s.dim, s.is_sparse, s.stream_alias, s['defines_mb_size']) for k, s in streams.items()]
    return cntk_py.ctf_deserializer(filename, sc)

def CBFDeserializer(filename, streams = {}):
    '''
    Configures the CNTK binary-format deserializer.

    Args:
        filename (str): file name containing the binary data
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          an input stream.

    See also:
        :cntkwiki:`CNTKBinaryReader format <BrainScript-CNTKBinary-Reader>`
    '''
    sc = [cntk_py.StreamConfiguration(
        k, s.dim, s.is_sparse, s.stream_alias) for k, s in streams.items()]
    return cntk_py.cbf_deserializer(filename, sc)

# TODO: this should be a private class; use StreamDef instead


class StreamConfiguration(cntk_py.StreamConfiguration):

    '''
    Configuration of a stream in a text format reader.

    Args:
        name (str): name of this stream
        dim (int): dimensions of this stream. A text format reader reads data
          as flat arrays. If you need different shapes you can
          :func:`~cntk.ops.reshape` it later.
        is_sparse (bool, defaults to `False`): whether the provided data is
          sparse (`False` by default)
        stream_alias (str, defaults to ''): name of the stream in the file
        defines_mb_size (`bool`, defaults to False): whether this stream defines
          the minibatch size.
    '''

    def __init__(self, name, dim, is_sparse=False, stream_alias='', defines_mb_size = False):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse,
                                                         stream_alias, defines_mb_size)

# stream definition for use in StreamDefs
# returns a record { stream_alias, is_sparse, optional shape, optional transforms, optional context, optional scp, optional mlf }
def StreamDef(field=None, shape=None, is_sparse=False, transforms=None,
              context=None, scp=None, mlf=None, broadcast=None, defines_mb_size=False, max_sequence_length = 65535):
    '''
       Configuration of a stream for use with the builtin Deserializers.
       The meanings of some configuration keys have a mild dependency on the
       exact deserializer, and certain keys are meaningless for certain
       deserializers.

    Args:
        field (`str`, defaults to `None`): this is the name of the stream

         * for CTFDeserializer the name is inside the CTF file
         * for ImageDeserializer the acceptable names are `image` or `label`
         * for HTKFeatureDeserializer and HTKMLFDeserializer only the default
           value of None is acceptable

        shape (`int` or `tuple`, defaults to `None`): dimensions of this
          stream. HTKFeatureDeserializer, HTKMLFDeserializer, and
          CTFDeserializer read data as flat arrays. If you need different
          shapes you can :func:`~cntk.ops.reshape` it later.
        is_sparse (`bool`, defaults to `False`): whether the provided data is
          sparse. `False` by default, unless mlf is provided.
        transforms (`list`, defaults to `None`): list of transforms to be
          applied by the Deserializer. Currently only ImageDeserializer
          supports transforms.
        context (`tuple`, defaults to `None`): left and right context to
          consider when reading in HTK data. Only supported by
          HTKFeatureDeserializer.
        scp (`str` or `list`, defaults to `None`): scp files for HTK data
        mlf (`str` or `list`, defaults to `None`): mlf files for HTK data
        broadcast (`bool`, defaults to `None`): whether the features in this
          stream should be broadcast to the whole sequence (useful in e.g.
          ivectors with HTK)
        defines_mb_size (`bool`, defaults to False): whether this stream defines
          the minibatch size.
        max_sequence_length (`int`, defaults to 65535): the upper limit on the length of consumed sequences. Sequence of larger size are skipped.
    '''
    config = dict(stream_alias=field, is_sparse=is_sparse)
    if shape is not None:
        config['dim'] = shape
    if transforms is not None:
        config['transforms'] = transforms
    if context is not None:
        config['context'] = context
    if scp is not None:
        config['scp'] = scp
    if mlf is not None:
        config['mlf'] = mlf
        config['is_sparse'] = True
    if broadcast is not None:
        config['broadcast'] = broadcast
    config['defines_mb_size'] = True if defines_mb_size else False
    config['max_sequence_length'] = max_sequence_length

    return Record(**config)
    # TODO: we should always use 'shape' unless it is always rank-1 or a single rank's dimension
    # TODO: dim should be inferred from the file, at least for dense


# StreamDefs for use in constructing deserializers
# StreamDefs(query = StreamDef(...), labels = StreamDef(...), ...)
StreamDefs = Record


def _dense_to_str(data):
    return ' '.join(data.ravel(order='C').astype(np.str))


def _sparse_to_str(data):
    return ' '.join('%s:%s' % (k, v) for k, v in sorted(data.items()))


def _is_tensor(data):
    '''
    Checks whether the data is a tensor, i.e. whether it is a NumPy array or a
    list of NumPy arrays.

    Args:
        data: data to check

    Returns:
      bool:
      `True`, if it is a tensor.
    '''
    if isinstance(data, np.ndarray):
        return True

    if not isinstance(data, list):
        return False

    while len(data) > 0:
        # All but the innermost dimension's values have to be lists
        try:
            data[0][0]
        except:
            # We reached the innermost dimension
            try:
                data[0] + 0
                return True
            except:
                # Innermost type is not a number
                return False

        if isinstance(data, np.ndarray):
            return True

        if not isinstance(data[0], list):
            return False

        data = data[0]

    return True


def sequence_to_cntk_text_format(seq_idx, alias_tensor_map):
    '''
    Converts a list of NumPy arrays representing tensors of inputs into a
    format that is readable by :class:`~cntk.io.CTFDeserializer`.

    Args:
        seq_idx (int): number of current sequence
        alias_tensor_map (dict): maps alias (str) to tensor (ndarray). Tensors
          are assumed to have dynamic axis.

    Returns:
        str:
        String representation in :cntkwiki:`CNTKTextReader format <BrainScript-CNTKTextFormat-Reader>`
    '''

    max_seq_length = max(len(t) for t in alias_tensor_map.values())

    if max_seq_length == 0:
        return ''

    lines = []
    for elem_idx in range(0, max_seq_length):
        line = []

        for alias, tensor in sorted(alias_tensor_map.items()):
            if elem_idx >= len(tensor):
                # for this alias there no more sequence elements
                continue

            if _is_tensor(tensor):
                if not isinstance(tensor, np.ndarray):
                    tensor = np.asarray(tensor)
                to_str = _dense_to_str
            elif isinstance(tensor, list) and isinstance(tensor[0], dict):
                to_str = _sparse_to_str
            else:
                raise ValueError(
                    'expected a tensor (dense) or list of dicts (sparse), but '
                    'got "%s"' % type(tensor))

            line.append('%s %s' % (alias, to_str(tensor[elem_idx])))

        lines.append('%i\t|' % seq_idx + ' |'.join(line))

    return '\n'.join(lines)

class UserDeserializer(cntk_py.SwigDataDeserializer):
    '''
    User deserializer is a base class for all user defined deserializers.
    To support deserialization of a custom format, please implement the public
    methods of this class and pass an instance of it to MinibatchSource.
    A UserDeserializer is a plug-in to MinibatchSource for reading data in custom formats.
    Reading data through this mechanism provides the following benefits:

      * randomization of data too large to fit into RAM, through CNTK chunked paging algorithm

      * distributed reading - only chunks needed by a particular worker are requested

      * composability of transforms (currently composability of user deserializers is not yet supported)

      * transparent support of sequence/frame/truncated BPTT modes

      * automatic chunk and minibatch prefetch

      * checkpointing

    The MinibatchSource uses the information provided by this class to build the timeline and move
    along it when the next minibatch is requested. The deserializer itself, however, is stateless.
    '''
    def __init__(self):
        super(UserDeserializer, self).__init__()
        self.__disown__()
        self._last_chunk = None

    def stream_infos(self):
        '''
        Should return a list of meta information :class:`StreamInformation` about all 
        streams exposed by the deserializer.

        Returns:
            list of :class:`StreamInformation` exposed by the deserializer
        '''
        raise NotImplementedError('should return a list of StreamInformation for all streams')

    def num_chunks(self):
        '''
        Should return the total number of chunks.
        '''
        raise NotImplementedError('should return the total number of chunks.')

    def get_chunk(self, chunk_id):
        '''
        Should return a dictionary of stream name -> data of the chunk, where data is csr_matrix/numpy array in sample mode,
        or a list of csr_matrix/numpy array in sequence mode.

        Args:
            chunk_id(int): id of the chunk to be read, 0 <= chunk_id < num_chunks

        Returns:
            dict containing the data
        '''
        raise NotImplementedError('should return data for the chunk.')

    def _stream_infos(self, infos=None):
        inner = self.stream_infos()
        if len(inner) == 0:
            raise ValueError('Deserializer must provide at least one stream')
        infos.extend(inner)

        streams = {si.m_name: si for si in inner}
        self.streams = Record(**streams)

    def _chunk_infos(self, infos=None):
        total = self.num_chunks()
        if total == 0:
            raise ValueError('Deserializer must provide at least one chunk')
        inner = []
        for i in range(total):
            t = cntk_py.ChunkInfo()
            t.m_id = i
            inner.append(t)
        infos.extend(inner)

    def _get_chunk(self, chunk_id):
        # Make sure the python object exists
        # till the next call, so that the copy in C++ can
        # take place.
        self._last_chunk = self.get_chunk(chunk_id=chunk_id)
        return self._last_chunk;
