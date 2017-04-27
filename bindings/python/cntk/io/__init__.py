# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import warnings
from .. import cntk_py, Value
from ..tensor import ArrayMixin
from cntk.internal import typemap, sanitize_dtype_cntk
from cntk.device import use_default_device
from cntk.logging import TraceLevel, get_trace_level
from cntk.variables import Record

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
    def data(self):
        '''
        The Value representation of the minibatch.
        '''
        return super(MinibatchData, self).data()

    @property
    def value(self):
        '''
        The value of the minibatch as a NumPy array.
        '''
        warnings.warn('the .value property is deprecated. Please use '
                      '.asarray() or .as_sequences() to get the NumPy '
                      'representations or .data to get the Value '
                      'representation', RuntimeWarning)

        return self.as_sequences()

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
        return self.data.mask().to_ndarray()

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
    MinibatchSource(deserializers, max_samples=cntk.io.INFINITELY_REPEAT, max_sweeps=cntk.io.INFINITELY_REPEAT, randomization_window_in_chunks=cntk.io.DEFAULT_RANDOMIZATION_WINDOW, randomization_window_in_samples=0, trace_level=cntk.logging.get_trace_level(), multithreaded_deserializer=False, frame_mode=False, truncation_length=0, randomize=None, randomization_window=None, sample_based_randomization_window=None, epoch_size=None)

    Args:
        deserializers (a single deserializer or a `list`): deserializers to be used in the composite reader
        max_samples (`int`, defaults to :const:`cntk.io.INFINITELY_REPEAT`): The maximum number of input samples
          (not 'label samples') the reader can produce. After this number has been reached, the reader
          returns empty minibatches on subsequent calls to :meth:`next_minibatch`. `max_samples` and `max_sweeps`
          are mutually exclusive, an exception will be raised if both have non-default values.
          **Important:**
          Click :cntkwiki:`here <BrainScript-epochSize-and-Python-epoch_size-in-CNTK>`
          for a description of input and label samples.
        max_sweeps (`int`, defaults to :const:`cntk.io.INFINITELY_REPEAT`): The maximum number of of sweeps over
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
        trace_level (an instance of :class:`cntk.logging.TraceLevel`): the output verbosity level, defaults to
          the current logging verbosity level given by :func:`~cntk.logging.get_trace_level`.
        multithreaded_deserializer (`bool`, defaults to `False`): specifies if the deserialization should be
          done on a single or multiple threads.
        frame_mode (`bool`, defaults to `False`): switches the frame mode on and off. If the frame mode
          is enabled the input data will be processed as individual frames ignoring all sequence information
          (this option cannot be used for BPTT, an exception will be raised if frame mode is enabled and the
          truncation length is non-zero).
        truncation_length (`int`, defaults to `0`): truncation length in samples, non-zero value enables
          the truncation (only applicable for BPTT, cannot be used in frame mode, an exception will be raised
          if frame mode is enabled and the truncation length is non-zero).
        randomize (`bool`, defaults to `None`): !DEPRECATED! please use randomization_window_in_chunks or
          randomization_window_in_samples instead
        randomization_window (int, defaults to `None`): !DEPRECATED! please use randomization_window_in_chunks or
          randomization_window_in_samples instead
        sample_based_randomization_window (`bool`, defaults to `None`): !DEPRECATED! please use
          randomization_window_in_chunks or randomization_window_in_samples instead
        epoch_size (`int`, defaults to `None`): !DEPRECATED! please use max_samples or max_sweeps instead
    '''
    def __init__(self,
        deserializers,
        max_samples = INFINITELY_REPEAT,
        max_sweeps = INFINITELY_REPEAT,
        randomization_window_in_chunks = DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
        randomization_window_in_samples = 0,
        trace_level = TraceLevel.Warning,
        multithreaded_deserializer=False,
        frame_mode=False,
        truncation_length=0,
        # all parameters below are deprecated
        randomize=None,
        randomization_window=None,
        sample_based_randomization_window=None,
        epoch_size=None,
        distributed_after=None):

        if not isinstance(deserializers, (list,tuple)):
            deserializers = [ deserializers ]

        config = cntk_py.MinibatchSourceConfig(deserializers)
        config.max_samples = max_samples
        config.max_sweeps = max_sweeps
        config.randomization_window_in_chunks = randomization_window_in_chunks
        config.randomization_window_in_samples = randomization_window_in_samples
        config.is_multithreaded = multithreaded_deserializer
        config.is_frame_mode_enabled = frame_mode
        config.truncation_length = truncation_length

        if isinstance(trace_level, TraceLevel):
            trace_level = trace_level.value

        config.trace_level = trace_level

        # the following deals with deprecated parameters.
        # TODO: 'randomize=False' is the only legacy option that still makes sense
        # (as a shortcut to randomization_window_in_chunks=0 and
        # randomization_window_in_samples=0), maybe we should keep it?
        if randomize is not None and randomize:
            warnings.warn('"randomize" parameter is deprecated and will be removed '
                'in future versions. Please specify "randomization_window_in_chunks" or '
                '"randomization_window_in_samples" instead', DeprecationWarning)
        elif randomize is None:
            randomize = True # previously default value

        if randomization_window is not None:
             warnings.warn('"randomization_window" parameter is deprecated and will be removed '
                'in future versions. Please specify "randomization_window_in_chunks" or '
                '"randomization_window_in_samples" instead', DeprecationWarning)
        else:
            randomization_window = DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS # previously default value

        if sample_based_randomization_window is not None:
             warnings.warn('"sample_based_randomization_window" parameter is deprecated and will be removed '
                'in future versions. Please specify "randomization_window_in_chunks" or '
                '"randomization_window_in_samples" instead', DeprecationWarning)
        else:
            sample_based_randomization_window = False  # previously default value

        if (randomize and sample_based_randomization_window):
            config.randomization_window_in_samples = randomization_window
            config.randomization_window_in_chunks = 0
        elif (randomize and not sample_based_randomization_window):
            config.randomization_window_in_chunks = randomization_window
            config.randomization_window_in_samples = 0
        elif not randomize:
            config.randomization_window_in_chunks = 0
            config.randomization_window_in_samples = 0

        if (epoch_size is not None):
            warnings.warn('"epoch_size" parameter is deprecated and will be removed '
                'in future versions. Please specify "max_samples" or '
                '"max_sweeps" instead', DeprecationWarning)
            config.max_samples = epoch_size

        source = cntk_py.create_composite_minibatch_source(config)
        # transplant into this class instance
        self.__dict__ = source.__dict__
        self._streams = None

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
        '''
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

        return {key: mb[value] for (key, value) in input_map.items()}

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            cntk.cntk_py.Dictionary:
            A :class:`~cntk.cntk_py.Dictionary` that has the checkpoint state
            of the MinibatchSource
        '''
        return super(MinibatchSource, self).get_checkpoint_state()

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (:class:`~cntk.cntk_py.Dictionary`): checkpoint to restore from
        '''
        super(MinibatchSource, self).restore_from_checkpoint(checkpoint)

    @property
    def is_distributed(self):
        '''
        Whether the minibatch source is running distributed
        '''
        return super(MinibatchSource, self).is_distributed()

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
    '''

    _storage = {'dense': cntk_py.StorageFormat_Dense,
                'sparse': cntk_py.StorageFormat_SparseCSC}

    def __init__(self, name, stream_id, storage_format, dtype,
                 shape):
        super(StreamInformation, self).__init__()
        self.m_name = name
        self.m_id = stream_id
        self.m_storage_format = StreamInformation._storage[storage_format]
        self.m_element_type = sanitize_dtype_cntk(dtype)
        self.m_sample_layout = cntk_py.NDShape(shape)


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

        Returns:
            mapping of :class:`StreamInformation` to :class:`MinibatchData`
        '''
        return NotImplementedError

    def _next_minibatch(self, info_map, mb_size_in_sequences,
            mb_size_in_samples, number_of_workers, worker_rank, device):
        # mbsize_in_sequences is ignored

        info_map.update(self.next_minibatch(mb_size_in_samples, device))

    def __getitem__(self, name):
        '''
        Return the :class:`StreamInformation` for the given
        stream name.

        Args:
            name (str): stream name to fetch
              :class:`StreamInformation` for
        '''
        return self.stream_info(name)


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
        left_context, right_context = stream.context if 'context' in stream\
                                                     else (0, 0)
        htk_config = cntk_py.HTKFeatureConfiguration(stream_name, scp_file,
                                                     dimension, left_context,
                                                     right_context, broadcast)
        feat.append(htk_config)

    if len(feat) == 0:
        raise ValueError("no feature streams found")
    return cntk_py.htk_feature_deserializer(feat)


def HTKMLFDeserializer(label_mapping_file, streams):
    '''
    Configures an HTK label reader that reads speech HTK format MLF (Master
    Label File)

    Args:
        label_mapping_file (str): path to the label mapping file
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          a label stream.
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
        return cntk_py.htk_mlf_deserializer(stream_name, label_mapping_file, dimension, master_label_files)


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
                "ImageDeserializer: invalid field name '{}', allowed are "
                "'image' and 'label'".format(alias))
    if image_stream_name is None:
        raise ValueError(
            "ImageDeserializer: stream name ('image' or 'label') must be "
            "specified")
    return cntk_py.image_deserializer(filename, label_stream_name, num_labels,
                                      image_stream_name, transforms)


def CTFDeserializer(filename, streams):
    '''
    Configures the CNTK text-format reader that reads text-based files with
    lines of the form::

        [Sequence_Id] (Sample)+

    where::

        Sample=|Input_Name (Value )*

    Args:
        filename (str): file name containing the text input

    See also:
        :cntkwiki:`CNTKTextReader format <BrainScript-CNTKTextFormat-Reader>`
    '''
    for k, s in streams.items():
        if s.stream_alias is None:
            raise ValueError("CTFDeserializer: stream name for key %s must be "
                             "specified" % k)
    sc = [cntk_py.StreamConfiguration(
        k, s.dim, s.is_sparse, s.stream_alias) for k, s in streams.items()]
    return cntk_py.ctf_deserializer(filename, sc)

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
    '''

    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse,
                                                         stream_alias)

# stream definition for use in StreamDefs
# returns a record { stream_alias, is_sparse, optional shape, optional transforms, optional context, optional scp, optional mlf }
def StreamDef(field=None, shape=None, is_sparse=False, transforms=None,
              context=None, scp=None, mlf=None, broadcast=None):
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
