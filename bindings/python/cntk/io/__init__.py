﻿# Copyright (c) Microsoft. All rights reserved.

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
        return self.m_num_sequences

    @property
    def num_samples(self):
        '''
        The number of samples in this minibatch
        '''
        return self.m_num_samples

    @property
    def value(self):
        '''
        The value of the minibatch as a NumPy array.
        '''
        return value_to_seq(self.m_data)

    @property
    def shape(self):
        '''
        The shape of the data in this minibatch as tuple.
        '''
        return self.m_data.shape().dimensions()

    @property
    def mask(self):
        '''
        The mask object of the minibatch. In it, `2` marks the beginning of a
        sequence, `1` marks a sequence element as valid, and `0` marks it as
        invalid.
        '''
        return self.m_data.mask().to_ndarray()

    @property
    def is_sparse(self):
        '''
        Whether the data in this minibatch is sparse.
        '''
        return self.m_data.is_sparse()

    def __len__(self):
        return self.num_sequences

class MinibatchSource(cntk_py.MinibatchSource):
    '''
    Parent class of all minibatch sources. For most cases you will need the
    helper functions :func:`text_format_minibatch_source` or
    :func:`minibatch_source`.
    A `MinibatchSource` can be indexed by the stream name, which will return a
    :class:`MinibatchData` object that can be passed e.g. to the
    :func:`~cntk.trainer.Trainer.train_minibatch` function.

    Args:
        deserializers ('list', default is empty): list of deserializers
         (:class:`ImageDeserializer` for now).
        randomize (bool, default True): randomize images before every epoch
        epoch_size (int): epoch size
        distributed_after (int): sample count after which minibatch source becomes distributed
        multithreaded_deserializer (bool): using multi threaded deserializer
    '''
    def __init__(self, deserializers=None, randomize=True, epoch_size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES, multithreaded_deserializer=None):
        if not isinstance(deserializers, (list,tuple)):
            deserializers = [deserializers] # allow passing a single item or a list
        reader_config = ReaderConfig(
            deserializers=deserializers,
            randomize=randomize,
            epoch_size=epoch_size,
            distributed_after=distributed_after,
            multithreaded_deserializer=multithreaded_deserializer)
        source = minibatch_source(reader_config)
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
            dict mapping input names to the stream information
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
        Return the :class:`StreamInfo` for the given stream name

        Args:
            name (str): stream name to fetch :class:`StreamInfo` for
        '''
        return self.stream_info(name)

    @typemap
    def next_minibatch(self, minibatch_size_in_samples,
            input_map=None, device=None):
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
            input_map (dict): mapping of :class:`~cntk.ops.variabls.Variable`
             to :class:`StreamInformation` which will be used to convert the
             returned data.
            device (`DeviceDescriptor`, defaults to `None`): CNTK DeviceDescriptor

        Returns:
            A mapping of :class:`StramInformation` to :class:`MinibatchData` if
            ``input_map`` was not specified. Otherwise, the returned value will
            be a mapping of :class:`~cntk.ops.variabls.Variable` to class:`MinibatchData`.
        '''
        if device is None:
            device = use_default_device()

        mb = super(MinibatchSource, self).get_next_minibatch(
                minibatch_size_in_samples, device)

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
            :class:`~cntk_py.Dictionary`
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
        :class:`~cntk_py.Dictionary`
    '''
    res = cntk_py.Dictionary()
    for k, v in py_dict.items():
        if isinstance(v, dict):
            res[k] = cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(v))
        # TODO: add support to list of lists ?
        elif isinstance(v, list):
            l = []
            for e in v:
                if isinstance(e, dict):
                    l.append(cntk_py.DictionaryValueFromDict(
                        _py_dict_to_cntk_dict(e)))
                else:
                    l.append(cntk_py.DictionaryValue(e))
            res[k] = cntk_py.DictionaryValue(l)
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res


# TODO: This should be a private function; use MinibatchSource(deserializer, ...).
@typemap
def minibatch_source(config):
    '''
    Instantiate the CNTK built-in composite minibatch source which is used to stream data into the network.
    Args:
        config (dict): a dictionary containing all the key-value configuration entries.
    Returns:
        :class:`MinibatchSource`
    '''
    cntk_dict = _py_dict_to_cntk_dict(config)
    return cntk_py.create_composite_minibatch_source(cntk_dict)

# TODO: This should be a private class.
class ReaderConfig(dict):
    '''
    Reader configuration.

    Args:
        deserializers ('list', default is empty): list of deserializers
         (:class:`ImageDeserializer` for now).
        randomize (bool, default True): randomize images before every epoch
        epoch_size (int): epoch size
        distributed_after (int): sample count after which reader becomes distributed
        multithreaded_deserializer (bool): using multi threaded deserializer
    '''
    def __init__(self, deserializers=None, randomize=True, epoch_size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES, multithreaded_deserializer=None):

        self['epochSize'] = cntk_py.SizeTWrapper(epoch_size) # force to store in size_t
        if not isinstance(deserializers, (list, tuple)):
            deserializers = [deserializers]
        self['deserializers'] = self.deserializers = deserializers or []
        self['randomize'] = randomize
        self['distributedAfterSampleCount'] = cntk_py.SizeTWrapper(distributed_after)
        if multithreaded_deserializer != None:
            self['multiThreadedDeserialization'] = multithreaded_deserializer

    @typemap
    def minibatch_source(self):
        '''
        Creates an instance of :class:`MinibatchSource` from this
        instance, which can be used to feed data into the `eval()` methods of
        the graph nodes or the `train_minibatch()` of :class:`~cntk.trainer.Trainer`.

        Returns:
            instance of :class:`MinibatchSource`
        '''
        return minibatch_source(self)


class Deserializer(dict):
    '''
    Base deserializer class that can be used in the :class:`ReaderConfig`. A
    deserializer is responsible for deserialization of input from external
    storage into in-memory sequences.

    Currently CNTK supports the below deserializers:

    ========================== ============
    Deserializer type          Description
    ========================== ============
    :class:`ImageDeserializer` Deserializer for images that uses OpenCV
    :class:`CTFDeserializer`   Deserializer for text of the `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/CNTKTextFormat-Reader>`_
    ========================== ============

    Args:
        type (str): type of the deserializer

    See also:
        https://github.com/microsoft/cntk/wiki/Understanding-and-Extending-Readers
    '''

    def __init__(self, type):
        self['type'] = type


class ImageDeserializer(Deserializer):
    '''
    This class configures the image reader that reads images and corresponding
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

    def __init__(self, filename, streams=None):
        super(ImageDeserializer, self).__init__('ImageDeserializer')
        self['file'] = filename
        self['input'] = self.input = {}
        # In ImageDeserializer, stream field names are hard-coded as "image" and "label".
        # These are configured in a somewhat inconsistent way.
        if streams is not None:
            for key in streams:
                s = streams[key]
                node = s.stream_alias
                if node == "image":
                    # BUGBUG: Can dim not be specified as well?
                    # TODO: clean this up and use a unified internal representation
                    self.map_features(key, s.transforms)
                elif node == "label":
                    self.map_labels(key, s.dim)
                else:
                    raise ValueError("ImageDeserializer: invalid field name '{}', allowed are 'image' and 'label'".format(node))

    # TODO: should be a private method; use constructor only
    def map_features(self, node, transforms):
        '''
        Maps feature node (either node instance or node name) to the transforms
        that will be applied to the images. It is usually applied to the input
        of the network with data augmentation.

        Args:
            node (str or input node): node or its name
            transforms (`list` of transforms): the transforms can be created by
             the static methods `crop`, `scale`, or `mean`.

        '''
        if not isinstance(node, str):
            node = node.name()
        if not isinstance(transforms, list):
            transforms = [transforms] if transforms else []
        self.input[node] = dict(transforms=transforms)

    # TODO: should be a private method; use constructor only
    def map_labels(self, node, num_classes):
        '''
        Maps label node (either node instance or node name)
        that will be applied to the images. It is usually used to define the
        ground truth of train or test.

        Args:
            node (str or input node): node or its name
            num_classes (int): number of classes

        '''
        if not isinstance(node, str):
            node = node.name()
        self.input[node] = dict(labelDim=num_classes) # reader distinguishes labels from features by calling this 'labelDim'

    @staticmethod
    def crop(crop_type='center', ratio=1.0, jitter_type='uniRatio'):
        '''
        Crop transform that can be used to pass to `map_features`

        Args:
            crop_type (str, default 'center'): 'center' or 'random'.  'random'
             is usually used during training while 'center' is usually for testing.
             Random cropping is a popular data augmentation technique used to improve
             generalization of the DNN.
            ratio (`float`, default 1.0): crop ratio. It specifies the ratio of
             final image dimension, e.g.  width , to the size of the random crop
             taken from the image. For example, the ratio 224 / 256 = 0.875 means
             crop of size 224 will be taken from the image rescaled to 256 (implementation
             detail:  ImageReader  takes the crop and then rescales instead of doing
             the other way around). To enable scale jitter (another popular data
             augmentation technique), use colon-delimited values like  cropRatio=0.875:0.466
             which means 224 crop will be taken from images randomly scaled to have
             size in [256, 480] range.
            jitter_type (str, default 'uniRatio'): crop scale jitter type, possible
             values are 'None', 'UniRatio'. 'uniRatio' means uniform distributed jitter
             scale between the minimum and maximum cropRatio values.

        Returns:
            dict describing the crop transform
        '''
        return dict(type='Crop', cropType=crop_type, cropRatio=ratio,
                jitterType=jitter_type)

    @staticmethod
    def scale(width, height, channels, interpolations='linear', scale_mode="fill", pad_value=-1):
        '''
        Scale transform that can be used to pass to `map_features` for data augmentation.

        Args:
            width (int): width of the image in pixels
            height (int): height of the image in pixels
            channels (int): channels of the image
            interpolations (str, default 'linear'): possible values are
             'nearest', 'linear', 'cubic', and 'lanczos'
            scale_mode (str, default 'fill'): 'fill', 'crop' or 'pad'.
             'fill' - warp the image to the given target size.
             'crop' - resize the image's shorter side to the given target size and crop the overlap.
             'pad'  - resize the image's larger side to the given target size, center it and pad the rest
            pad_value (int, default -1): -1 or int value. The pad value used for the 'pad' mode.
             If set to -1 then the border will be replicated.

        Returns:
            dict describing the scale transform
        '''
        return dict(type='Scale', width=width, height=height, channels=channels,
                interpolations=interpolations, scaleMode=scale_mode, padValue=pad_value)

    @staticmethod
    def mean(filename):
        '''
        Mean transform that can be used to pass to `map_features` for data augmentation.

        Args:
            filename (str): file that stores the mean values for each pixel
             in OpenCV matrix XML format

        Returns:
            dict describing the mean transform
        '''
        return dict(type='Mean', meanFile=filename)

    # TODO color transpose


class CTFDeserializer(Deserializer):
    '''
    This class configures the text reader that reads text-encoded files from a
    file with lines of the form::

        [Sequence_Id](Sample)+

    where::

        Sample=|Input_Name (Value )*

    Args:
        filename (str): file name containing the text input

    See also:
        `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/CNTKTextFormat-Reader>`_
    '''

    def __init__(self, filename, streams=None):
        super(CTFDeserializer, self).__init__('CNTKTextFormatDeserializer')
        self['file'] = filename
        self['input'] = self.input = {}
        # connect all streams (: StreamDef) if given
        if streams is not None:
            for key in streams:
                s = streams[key]
                # TODO: guard against any other fields, such as transformers, which is not valid here
                self.map_input(key, s.dim, "sparse" if s.is_sparse else "dense", alias=s.stream_alias)

    # TODO: should be a private method; use constructor only
    def map_input(self, node, dim, format="dense", alias=None):
        '''
        Maps node (either node instance or node name) to a part of the text input,
        either specified by the node name or the alias in the text file.

        Example: for node name 'input0' an input line could look like this::

          |input0 3 7 1 0 2

        Args:
            node (str or input node): node or its name
            dim (int): specifies the dimension of the input value vector
             (for dense input this directly corresponds to the number of values in each sample,
             for sparse this represents the upper bound on the range of possible index values).
            format (str, default 'dense'): 'dense' or 'sparse'. Specifies the input type.
            alias (str, default None): None or alias name. Optional abbreviated name that
             is used in the text file to avoid repeating long input names. For details please
             see `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/CNTKTextFormat-Reader>`_
        '''
        if not isinstance(node, str):
            node = node.name()
        if alias is None:
            alias=node
        self.input[node] = dict(dim=dim, format=format, alias=alias)


# TODO: This should not exist; use MinibatchSource(CTFDeserializer(...))
@typemap
def text_format_minibatch_source(path, stream_configs, epoch_size=INFINITELY_REPEAT, randomize=True, distributed_after=INFINITE_SAMPLES):
    '''
    Creates a minibatch source from a CNTKTextFormatReader file.

    Args:
        path (file): filename of the data file
        stream_configs (`list` of :class:`StreamConfiguration` instances): list
         of stream configurations, each of which describes one stream in the
         file
        epoch_size (int, optional): size of an epoch. In case of 0 the size
         of the training set will be taken. Default is max of 64bit.
        randomize (bool, optional): whether to randomize the contents of data file.
        distributed_after (int, optional): sample count after which minibatch source becomes distributed

    Returns:
        :class:`MinibatchSource`
    '''
    return cntk_py.text_format_minibatch_source(path, stream_configs, epoch_size, randomize, distributed_after)


# TODO: this should be a private class; use StreamDef instead
class StreamConfiguration(cntk_py.StreamConfiguration):
    '''
    Configuration of a stream in a text format reader. This can be used in
    :func:`text_format_minibatch_source`.

    Args:
        name (str): name of this stream
        dim (int): dimensions of this stream. A text format reader reads data
         as flat arrays. If you need different shapes you can
         :func:`~cntk.ops.reshape` it later.
        is_sparse (bool, default `False`): whether the provided data is sparse
         (`False` by default)
        stream_alias (str, default ''): name of the stream in the file that is fed to the
         :func:`text_format_minibatch_source`
    '''

    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse, stream_alias)


# wrapper around text_format_minibatch_source() that attaches a record of streams
# TODO: This should not exist; use MinibatchSource(CTFDeserializer(...))
def _unused_CNTKTextFormatMinibatchSource(path, streams, epoch_size=None): # TODO: delete this
    from cntk.utils import _ClassFromDict
    # convert streams into StreamConfiguration format
    # TODO: stream_alias should default to 'key'
    stream_configs = [ StreamConfiguration(key, dim=value.dim, is_sparse=value.is_sparse, stream_alias=value.stream_alias) for (key, value) in streams.items() ]
    if epoch_size is not None:  # TODO: use MAX_UI64, now that we have access
        source = text_format_minibatch_source(path, stream_configs, epoch_size)
    else:
        source = text_format_minibatch_source(path, stream_configs)
    # attach a dictionary of the streams
    source.streams = _ClassFromDict({ name : source.stream_info(name) for name in streams.keys() })
    return source


# stream definition for use in StreamDefs
# returns a record { stream_alias, is_sparse, optional dim, optional transforms }
from cntk.utils import Record
def StreamDef(field, shape=None, is_sparse=False, transforms=None):
    # note: the names used inside here are required by the C++ code which looks them up in a dictionary
    config = dict(stream_alias=field, is_sparse=is_sparse)
    if shape is not None:
        config['dim'] = shape
    if transforms is not None:
        config['transforms'] = transforms
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

    Returns: True, if it is a tensor.
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


