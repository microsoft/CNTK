# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py
from ..tensor import ArrayMixin
from ..utils import typemap, value_to_seq
from cntk.device import use_default_device

import numpy as np

INFINITELY_REPEAT = cntk_py.MinibatchSource.infinitely_repeat
FULL_DATA_SWEEP = cntk_py.MinibatchSource.full_data_sweep
INFINITE_SAMPLES = cntk_py.MinibatchSource.infinite_samples
DEFAULT_RANDOMIZATION_WINDOW = cntk_py.MinibatchSource.default_randomization_window
DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS = cntk_py.MinibatchSource.default_randomization_window_in_chunks

class MinibatchData(cntk_py.MinibatchData, ArrayMixin):
    '''
    Holds a minibatch of input data. This is never directly created, but
    only returned by :class:`MinibatchSource` instances.
    '''

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

    @property
    def value(self):
        '''
        The value of the minibatch as a NumPy array.
        '''
        return value_to_seq(self.data)

    @property
    def shape(self):
        '''
        The shape of the data in this minibatch as tuple.
        '''
        return self.data.shape().dimensions()

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
        return self.data.is_sparse()

    def __len__(self):
        return self.num_sequences

class MinibatchSource(cntk_py.MinibatchSource):
    '''MinibatchSource(deserializers=None, randomize=True, randomization_window=DEFAULT_RANDOMIZATION_WINDOW, epoch_size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES, multithreaded_deserializer=None)
    A `MinibatchSource` can be indexed by the stream name, which will return a
    Parent class of all minibatch sources.  A `MinibatchSource` can be indexed by the stream name, which will return a
    :class:`MinibatchData` object that can be passed e.g. to the
    :func:`~cntk.trainer.Trainer.train_minibatch` function.

    Args:
        deserializers (`list`, default to empty): list of deserializers
        randomize (`bool`, default to `True`): randomize before every epoch
        randomization_window (int): size of window that reader will shuffle, ignored if `randomize`
          is `False`
        sample_based_randomization_window (`bool`, default to `False`): specifies how to interpret
          `randomization_window`. If `True`, the size of the randomization window is interpreted as a certain
          number of samples, otherwise -- as a number of chunks. Similarly to `randomization_window`,
          this parameter is ignored, when `randomize` is `False`
        epoch_size (`int`, default to `INFINITELY_REPEAT`): number of samples as a scheduling unit.
          Parameters in the schedule change their values every `epoch_size`
          samples. If no `epoch_size` is provided, this parameter is substituted
          by the size of the full data sweep with infinte repeat, in which case the scheduling unit is
          the entire data sweep (as indicated by the MinibatchSource) and parameters
          change their values on the sweep-by-sweep basis specified by the schedule.
        distributed_after (int, default to `INFINITE_SAMPLES`): sample count after which minibatch source becomes distributed
        multithreaded_deserializer (`bool`, default to `None`): using multi threaded deserializer
    '''
    def __init__(self,
        deserializers=None,
        randomize=True,
        randomization_window=DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
        sample_based_randomization_window=False,
        epoch_size=INFINITELY_REPEAT,
        distributed_after=INFINITE_SAMPLES,
        multithreaded_deserializer=None):

        if not isinstance(deserializers, (list,tuple)):
            deserializers = [deserializers] # allow passing a single item or a list
        reader_config = _ReaderConfig(
            deserializers=deserializers,
            randomize=randomize,
            randomization_window=randomization_window,
            sample_based_randomization_window=sample_based_randomization_window,
            epoch_size=epoch_size,
            distributed_after=distributed_after,
            multithreaded_deserializer=multithreaded_deserializer)
        source = reader_config.minibatch_source()
        # transplant into this class instance
        self.__dict__ = source.__dict__
        # transplant all members of deserializers into a record called streams
        streams = {}
        for si in self.stream_infos():
            streams[si.m_name] = si
        from ..utils import Record
        self.streams = Record(**streams)

    def stream_infos(self):
        '''
        Describes the stream that this source produces.

        Returns:
            dict:
            A `dict` mapping input names to the stream information
        '''
        return super(MinibatchSource, self).stream_infos()

    def stream_info(self, name):
        '''
        Gets the description of the stream with given name.
        Throws an exception if there are none or multiple streams with this
        same name.
        '''
        return super(MinibatchSource, self).stream_info(name)

    def __getitem__(self, name):
        '''
        Return the :class:`~cntk.cntk_py.StreamInformation` for the given stream name

        Args:
            name (str): stream name to fetch :class:`~cntk.cntk_py.StreamInformation` for
        '''
        return self.stream_info(name)

    @typemap
    def next_minibatch(self, minibatch_size_in_samples,
            input_map=None, device=None, num_data_partitions=None, partition_index=None):
        '''
        Reads a minibatch that contains data for all input streams.  The
        minibatch size is specified in terms of #samples and/or #sequences for the
        primary input stream; value of 0 for #samples/#sequences means
        unspecified.  In case the size is specified in terms of both #sequences
        and #samples, the smaller of the 2 is taken.  An empty map is returned
        when the MinibatchSource has no more data to return.

        Args:
            minibatch_size_in_samples (int): number of samples to retrieve for
              the next minibatch. Must be > 0.
            input_map (dict): mapping of :class:`~cntk.ops.variables.Variable`
              to :class:`~cntk.cntk_py.StreamInformation` which will be used to convert the
              returned data.
            device (`DeviceDescriptor`, defaults to `None`): CNTK DeviceDescriptor
            num_data_partitions: Used for distributed training, indicates into how many partitions
              the source should split the data.
            partition_index (`int`, defaults to `None`): Used for distributed training, indicates data from which partition to take.

        Returns:
            cntk.io.MinibatchData:
            A mapping of :class:`~cntk.cntk_py.StreamInformation` to :class:`MinibatchData` if
            `input_map` was not specified. Otherwise, the returned value will
            be a mapping of :class:`~cntk.ops.variables.Variable` to class:`MinibatchData`.
        '''
        if device is None:
            device = use_default_device()

        if num_data_partitions is None:
            num_data_partitions = 1

        if partition_index is None:
            partition_index = 0

        mb = super(MinibatchSource, self).get_next_minibatch(0,
                minibatch_size_in_samples, num_data_partitions, partition_index, device)

        if input_map:
            if not mb:
                return {}
            else:
                return { key : mb[value] for (key, value) in input_map.items() }
        else:
            return mb

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            cntk.cntk_py.Dictionary:
            A :class:`~cntk.cntk_py.Dictionary` that has the checkpoint state of the MinibatchSource
        '''
        return super(MinibatchSource, self).get_checkpoint_state()

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (:class:`~cntk_py.Dictionary`): checkpoint to restore from
        '''
        super(MinibatchSource, self).restore_from_checkpoint(checkpoint)

    @property
    def is_distributed(self):
        '''
        Whether the minibatch source is running distributed
        '''
        return super(MinibatchSource, self).is_distributed()

def _py_dict_to_cntk_dict(py_dict):
    '''
    Converts a Python dictionary into a CNTK Dictionary whose values are CNTK DictionaryValue instances.

    Args:
        py_dict (dict): a dictionary to be converted.

    Returns:
        cntk_py.Dictionary:
        A :class:`~cntk_py.Dictionary` that has been converted from the input `dict`
    '''
    res = cntk_py.Dictionary()
    for k, v in py_dict.items():
        if isinstance(v, dict):
            res[k] = cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(v))
        # TODO: add support to list of lists ?
        elif isinstance(v, list):
            res[k] = cntk_py.DictionaryValue([cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(e) if isinstance(e, dict) else e) for e in v])
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res


# TODO: This should be a private function; use MinibatchSource(deserializer, ...).
@typemap
def _minibatch_source(config):
    '''
    Instantiate the CNTK built-in composite minibatch source which is used to stream data into the network.

    Args:
        config (dict): a dictionary containing all the key-value configuration entries.

    Returns:
        cntk.io.MinibatchSource:
        The :class:`MinibatchSource` used to stream data into the network
    '''
    cntk_dict = _py_dict_to_cntk_dict(config)
    return cntk_py.create_composite_minibatch_source(cntk_dict)

class _ReaderConfig(dict):
    '''
    Reader configuration.

    Args:
        deserializers ('list', default is `None`): list of deserializers
         (:class:`ImageDeserializer` for now).
        randomize (`bool`, default to `True`): randomize images before every epoch
        randomization_window (int): size of window that reader will shuffle, ignored if `randomize`
          is `False`
        sample_based_randomization_window (bool, default False): specifies how to interpret
          `randomization_range`. If `True`, the size of the randomization window is interpreted as a certain
          number of samples, otherwise -- as a number of chunks. Similarly to `randomization_window`,
          this parameter is ignored, when `randomize` is `False`
        epoch_size (`int`, default to `INFINITELY_REPEAT`): number of samples as a scheduling unit.
          Parameters in the schedule change their values every `epoch_size`
          samples. If no `epoch_size` is provided, this parameter is substituted
          by the size of the full data sweep with infinte repeat, in which case the scheduling unit is
          the entire data sweep (as indicated by the MinibatchSource) and parameters
          change their values on the sweep-by-sweep basis specified by the schedule.
        distributed_after (int, default to `INFINITE_SAMPLES`): sample count after which reader becomes distributed
        multithreaded_deserializer (`bool`, default to `None`): using multi threaded deserializer
    '''
    def __init__(self,
        deserializers=None,
        randomize=True,
        randomization_window=DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
        sample_based_randomization_window=False,
        epoch_size=INFINITELY_REPEAT,
        distributed_after=INFINITE_SAMPLES,
        multithreaded_deserializer=None):
        self['epochSize'] = cntk_py.SizeTWrapper(epoch_size) # force to store in size_t
        if not isinstance(deserializers, (list, tuple)):
            deserializers = [deserializers]
        self['deserializers'] = self.deserializers = deserializers or []
        self['randomize'] = randomize
        self['randomizationWindow'] = cntk_py.SizeTWrapper(randomization_window)
        self['sampleBasedRandomizationWindow'] = sample_based_randomization_window
        self['distributedAfterSampleCount'] = cntk_py.SizeTWrapper(distributed_after)
        if multithreaded_deserializer is not None:
            self['multiThreadedDeserialization'] = multithreaded_deserializer

    @typemap
    def minibatch_source(self):
        '''
        Creates an instance of :class:`MinibatchSource` from this
        instance, which can be used to feed data into the `eval()` methods of
        the graph nodes or the `train_minibatch()` of :class:`~cntk.trainer.Trainer`.

        Returns:
            cntk.io.MinibatchSource:
            An instance of :class:`MinibatchSource` from this instance.
        '''
        return _minibatch_source(self)

def HTKFeatureDeserializer(streams):
    '''
    Configures the HTK feature reader that reads speech data from scp files.

    Args:
        streams: any dictionary-like object that contains a mapping from stream names
            to :class:`StreamDef` objects. Each StreamDef object configures a feature stream.
    '''
    feat = []
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None: raise ValueError("HTKFeatureDeserializer does not support steam names")
        if 'scp' not in stream: raise ValueError("No scp files specified for HTKFeatureDeserializer")
        dimension = stream.dim
        scp_file = stream['scp']
        broadcast = stream['broadcast'] if 'broadcast' in stream else False
        left_context, right_context = stream.context if 'context' in stream else (0, 0)
        feat.append(cntk_py.HTKFeatureConfiguration(stream_name, scp_file, dimension, left_context, right_context, broadcast))
    if len(feat) == 0:
        raise ValueError("no feature streams found")
    return cntk_py.htk_feature_deserializer(feat)

def HTKMLFDeserializer(label_mapping_file, streams):
    '''
    Configures an HTK label reader that reads speech HTK format MLF (Master Label File)

    Args:
        label_mapping_file (str): path to the label mapping file
        streams: any dictionary-like object that contains a mapping from stream names
            to :class:`StreamDef` objects. Each StreamDef object configures a label stream.
    '''
    if len(streams) != 1: raise ValueError("HTKMLFDeserializer only accepts a single stream")
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None: raise ValueError("HTKMLFDeserializer does not support steam names")
        dimension = stream.dim
        if 'mlf' not in stream: raise ValueError("No master label files specified for HTKMLFDeserializer")
        master_label_files = stream['mlf']
        if not isinstance(master_label_files,list):
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
        `Image reader definition <https://github.com/microsoft/cntk/wiki/Image-reader>`_
    '''
    image_stream_name = None
    label_stream_name = '_ignore_labels_'
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
            raise ValueError("ImageDeserializer: invalid field name '{}', allowed are 'image' and 'label'".format(alias))
    if image_stream_name is None:
        raise ValueError("ImageDeserializer: stream name ('image' or 'label') must be specified")
    return cntk_py.image_deserializer(filename, label_stream_name, num_labels, image_stream_name, transforms)

def CTFDeserializer(filename, streams):
    '''
    Configures the CNTK text-format reader that reads text-based files with lines of the form::

        [Sequence_Id] (Sample)+

    where::

        Sample=|Input_Name (Value )*

    Args:
        filename (str): file name containing the text input

    See also:
        `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/CNTKTextFormat-Reader>`_
    '''
    for k,s in streams.items():
        if s.stream_alias is None:
            raise ValueError("CTFDeserializer: stream name for key %s must be specified"%(k))
    sc = [cntk_py.StreamConfiguration(k, s.dim, s.is_sparse, s.stream_alias) for k,s in streams.items()]
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
        is_sparse (bool, default `False`): whether the provided data is sparse
          (`False` by default)
        stream_alias (str, default ''): name of the stream in the file
    '''
    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse, stream_alias)

# stream definition for use in StreamDefs
# returns a record { stream_alias, is_sparse, optional shape, optional transforms, optional context, optional scp, optional mlf }
from cntk.utils import Record
def StreamDef(field=None, shape=None, is_sparse=False, transforms=None, context=None, scp=None, mlf=None, broadcast=None):
    '''
       Configuration of a stream for use with the builtin Deserializers.
       The meanings of some configuration keys have a mild dependency on the
       exact deserializer, and certain keys are meaningless for certain deserializers.

    Args:
        field (str): this is the name of the stream:
        
         * for CTFDeserializer the name is inside the CTF file
         * for ImageDeserializer the acceptable names are `image` or `label`
         * for HTKFeatureDeserializer and HTKMLFDeserializer only the default 
           value of None is acceptable
        
        shape (int, tuple): dimensions of this stream. HTKFeatureDeserializer, 
         HTKMLFDeserializer, and CTFDeserializer read data
         as flat arrays. If you need different shapes you can
         :func:`~cntk.ops.reshape` it later.
        is_sparse (bool): whether the provided data is sparse.
         `False` by default, unless mlf is provided.
        transforms (list): list of transforms to be applied by the Deserializer. 
         Currently only ImageDeserializer supports transforms.
        context (tuple): left and right context to consider when reading in HTK 
         data. Only supported by HTKFeatureDeserializer.
        scp (str, list): scp files for HTK data
        mlf (str, list): mlf files for HTK data
        broadcast (bool): whether the features in this stream should be 
         broadcast to the whole sequence (useful in e.g. ivectors with HTK)
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
        String representation in `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/CNTKTextFormat-Reader>`_
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
                    'expected a tensor (dense) or list of dicts (sparse), but got "%s"' % type(tensor))

            line.append('%s %s' % (alias, to_str(tensor[elem_idx])))

        lines.append('%i\t|' % seq_idx + ' |'.join(line))

    return '\n'.join(lines)


