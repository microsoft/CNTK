# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py
from ..utils import typemap
from cntk.device import use_default_device
MAX_UI64 = int('0xffffffffffffffff', 16)

class MinibatchData(cntk_py.MinibatchData):
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
        from ..utils import value_to_seq
        return value_to_seq(self.m_data)

    @property
    def mask(self):
        '''
        The mask object of the minibatch. In it, `2` marks the beginning of a
        sequence, `1` marks a sequence element as valid, and `0` markse it as
        invalid.
        '''
        return self.m_data.mask().to_numpy()

    def __len__(self):
        return self.num_sequences

class MinibatchSource(cntk_py.MinibatchSource):
    '''
    Parent class of all minibatch sources. For most cases you will need the
    helper functions :func:`text_format_minibatch_source` or
    :func:`minibatch_source`.
    A `MinibatchSource` can be indexed by a `StreamInfo`, which will return a
    `MinibatchData` object that can be passed e.g. to the
    :func:`cntk.trainer.Trainer.train_minibatch` function.
    '''

    def stream_infos(self):
        '''
        Describes the stream that this source produces.

        Returns:
            `dict` mapping input names to the stream information
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
            name (`str`): stream name to fetch :class:`StreamInfo` for
        '''
        return self.stream_info(name)

    @typemap
    def next_minibatch(self, minibatch_size_in_samples=None,
            minibatch_size_in_sequences=None, input_map=None,
            device=None):
        '''
        Reads a minibatch that contains data for all input streams.  The
        minibatch size is specified in terms of #samples and/or #sequences for the
        primary input stream; value of 0 for #samples/#sequences means
        unspecified.  In case the size is specified in terms of both #sequences
        and #samples, the smaller of the 2 is taken.  An empty map is returned
        when the MinibatchSource has no more data to return.

        Args:
            minibatch_size_in_samples (`int`): number of samples to retrieve for
             the next minibatch. Must be > 0.
            minibatch_size_in_sequences (`int`, defaults to `None`): number of
             samples to retrieve for the next minibatch. Must be > 0.
            input_map (`dict`): mapping of :class:`cntk.ops.variabls.Variable`
             to :class:`StreamInformation` which will be used to convert the
             returned data.
            device (`DeviceDescriptor`, defaults to `None`): CNTK DeviceDescriptor

        Returns:
            A mapping of :class:`StramInformation` to :class:`MinibatchData` if
            ``input_map`` was not specified. Otherwise, the returned value will
            be a mapping of :class:`cntk.ops.variabls.Variable` to class:`MinibatchData`.
        '''
        if device is None:
            device = use_default_device()

        if minibatch_size_in_samples is None and \
                minibatch_size_in_sequences is None:
            raise ValueError('you have to specify at least one of '
                    'minibatch_size_in_samples or minibatch_size_in_sequences')

        if minibatch_size_in_sequences is None:
            mb = super(MinibatchSource, self).get_next_minibatch(
                minibatch_size_in_samples, device)
        else:
            if minibatch_size_in_samples is None:
                minibatch_size_in_samples = 0

            mb = super(MinibatchSource, self).get_next_minibatch(
                minibatch_size_in_samples,
                minibatch_size_in_sequences, device)

        if input_map:
            if not mb:
                return None
            else:
                return { key : mb[value] for (key, value) in input_map.items() }
        else:
            return mb

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            :class:`cntk_py.Dictionary`
        '''
        return super(MinibatchSource, self).get_checkpoint_state()

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (:class:`cntk_py.Dictionary`): checkpoint to restore from
        '''
        super(MinibatchSource, self).restore_from_checkpoint(checkpoint)


def _py_dict_to_cntk_dict(py_dict):
    '''
    Converts a Python dictionary into a CNTK Dictionary whose values are CNTK DictionaryValue instances.
    Args:
        py_dict (`dict`): a dictionary to be converted.
    Returns:
        :class:`cntk_py.Dictionary`
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
                    l.append(cntk_py.DictionaryValue(v))
            res[k] = cntk_py.DictionaryValue(l)
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res


@typemap
def minibatch_source(config):
    '''
    Instantiate the CNTK built-in composite minibatch source which is used to stream data into the network.
    Args:
        config (`dict`): a dictionary containing all the key-value configuration entries.
    Returns:
        :class:`MinibatchSource`
    '''
    cntk_dict = _py_dict_to_cntk_dict(config)
    return cntk_py.create_composite_minibatch_source(cntk_dict)


class ReaderConfig(dict):
    '''
    Reader configuration.

    Args:
        deserializers ('list', default is empty): list of deserializers
         (:class:`ImageDeserializer` for now).
        randomize (`bool`, default True): randomize images before every epoch
        epoch_size (`int`): epoch size
    '''

    def __init__(self, deserializers=None, randomize=True, epoch_size=MAX_UI64):

        self['epochSize'] = epoch_size
        if not isinstance(deserializers, (list, tuple)):
            deserializers = [deserializers]
        self['deserializers'] = self.deserializers = deserializers or []
        self['randomize'] = randomize

    @typemap
    def minibatch_source(self):
        '''
        Creates an instance of :class:`MinibatchSource` from this
        instance, which can be used to feed data into the `eval()` methods of
        the graph nodes or the `train_minibatch()` of :class:`cntk.trainer.Trainer`.

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
    ========================== ============

    Args:
        type (`str`): type of the deserializer

    See also:
        https://github.com/microsoft/cntk/wiki/Understanding-and-Extending-Readers
    '''

    def __init__(self, type):
        self['type'] = type


class ImageDeserializer(Deserializer):
    '''
    This class configures the image reader that reads images and corresponding
    labels from a file of the form

         <full path to image><tab><numerical label (0-based class id)>

    Args:
        filename (`str`): file name of the map file that associates images to
         classes

    See also:
        https://github.com/microsoft/cntk/wiki/Image-reader
    '''

    def __init__(self, filename):
        super(ImageDeserializer, self).__init__('ImageDeserializer')
        self['file'] = filename
        self['input'] = self.input = {}

    def map_features(self, node, transforms):
        '''
        Maps feature node (either node instance or node name) to the transforms
        that will be applied to the images. It is usually applied to the input
        of the network with data augmentation.

        Args:
            node (`str` or input node): node or its name
            transforms (`list` of transforms): the transforms can be created by
             the static methods `crop`, `scale`, or `mean`.

        '''
        if not isinstance(node, str):
            node = node.name()
        if not isinstance(transforms, list):
            transforms = [transforms] if transforms else []
        self.input[node] = dict(transforms=transforms)

    def map_labels(self, node, num_classes):
        '''
        Maps label node (either node instance or node name)
        that will be applied to the images. It is usually used to define the
        ground truth of train or test.

        Args:
            node (`str` or input node): node or its name
            num_classes (`int`): number of classes

        '''
        if not isinstance(node, str):
            node = node.name()
        self.input[node] = dict(labelDim=num_classes)

    @staticmethod
    def crop(crop_type='center', ratio=1.0, jitter_type='uniRatio'):
        '''
        Crop transform that can be used to pass to `map_features`

        Args:
            crop_type (`str`, default 'center'): 'center' or 'random'.  'random'
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
            jitter_type (`str`, default 'uniRatio'): crop scale jitter type, possible
             values are 'None', 'UniRatio'. 'uniRatio' means uniform distributed jitter
             scale between the minimum and maximum cropRatio values.

        Returns:
            `dict` describing the crop transform
        '''
        return dict(type='Crop', cropType=crop_type, cropRatio=ratio,
                jitterType=jitter_type)

    @staticmethod
    def scale(width, height, channels, interpolations='linear'):
        '''
        Scale transform that can be used to pass to `map_features` for data augmentation.

        Args:
            width (`int`): width of the image in pixels
            height (`int`): height of the image in pixels
            channels (`int`): channels of the image
            interpolations (`str`, default 'linear'): possible values are
             'nearest', 'linear', 'cubic', and 'lanczos'

        Returns:
            `dict` describing the scale transform
        '''
        return dict(type='Scale', width=width, height=height, channels=channels,
                interpolations=interpolations)

    @staticmethod
    def mean(filename):
        '''
        Mean transform that can be used to pass to `map_features` for data augmentation.

        Args:
            filename (`str`): file that stores the mean values for each pixel
             in OpenCV matrix XML format

        Returns:
            `dict` describing the mean transform
        '''
        return dict(type='Mean', meanFile=filename)

    # TODO color transpose

#
# CNTKTextFormatReader
# TODO get away from cntk_py.text_format_minibatch_source and set it up
# similarly to ImageDeserializer
#


@typemap
def text_format_minibatch_source(path, stream_configs, epoch_size=MAX_UI64, randomize=True):
    '''
    Creates a minibatch source from a CNTKTextFormatReader file.

    Args:
        path ('file'): filename of the data file
        stream_configs (`list` of :class:`StreamConfiguration` instances): list
         of stream configurations, each of which describes one stream in the
         file
        epoch_size (`int`, optional): size of an epoch. In case of 0 the size
         of the training set will be taken. Default is max of 64bit.
        randomize (`bool`, optional): whether to randomize the contents of data file.

    Returns:
        :class:`MinibatchSource`
    '''
    return cntk_py.text_format_minibatch_source(path, stream_configs,
                                                epoch_size, randomize)


class StreamConfiguration(cntk_py.StreamConfiguration):
    '''
    Configuration of a stream in a text format reader. This can be used in
    :func:`text_format_minibatch_source`.

    Args:
        name (`str`): name of this stream
        dim (`int`): dimensions of this stream. A text format reader reads data
         as flat arrays. If you need different shapes you can
         :func:`cntk.ops.reshape` it later.
        is_sparse (`bool`, default `False`): whether the provided data is sparse
         (`False` by default)
        stream_alias (`str`, default ''): name of the stream in the file that is fed to the
         :func:`text_format_minibatch_source`
    '''

    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse, stream_alias)


# wrapper around text_format_minibatch_source() that attaches a record of streams
def CNTKTextFormatMinibatchSource(path, streams, epoch_size=None):
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


# stream definition for CNTKTextFormatMinibatchSource
def StreamDef(shape, is_sparse, alias):
    from cntk.utils import Record
    return Record(dim=shape, is_sparse=is_sparse, stream_alias=alias)
    # TODO: why stream_alias and not alias?
    # TODO: we should always use 'shape' unless it is always rank-1 or a single rank's dimension
    # TODO: dim should be inferred from the file, at least for dense
