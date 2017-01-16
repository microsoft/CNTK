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
        Indicates whether the data in this minibatch is comes from a sweep end
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
    '''
    Parent class of all minibatch sources. For most cases you will need the
    helper functions :func:`text_format_minibatch_source` or
    :func:`minibatch_source`.
    A `MinibatchSource` can be indexed by the stream name, which will return a
    :class:`MinibatchData` object that can be passed e.g. to the
    :func:`~cntk.trainer.Trainer.train_minibatch` function.

    Args:
        deserializers ('list', default is empty): list of deserializers
        randomize (bool, default True): randomize before every epoch
        randomization_window (int) : size of window that reader will shuffle, ignored if `randomize` is False
        epoch_size (int): epoch size
        distributed_after (int): sample count after which minibatch source becomes distributed
        multithreaded_deserializer (bool): using multi threaded deserializer
    '''
    def __init__(self, deserializers=None, randomize=True, randomization_window=DEFAULT_RANDOMIZATION_WINDOW, epoch_size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES, multithreaded_deserializer=None):
        if not isinstance(deserializers, (list,tuple)):
            deserializers = [deserializers] # allow passing a single item or a list
        reader_config = ReaderConfig(
            deserializers=deserializers,
            randomize=randomize,
            randomization_window=randomization_window,
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
        randomization_window (int) : size of window that reader will shuffle, ignored if `randomize` is False
        epoch_size (int): epoch size
        distributed_after (int): sample count after which reader becomes distributed
        multithreaded_deserializer (bool): using multi threaded deserializer
    '''
    def __init__(self, deserializers=None, randomize=True, randomization_window=DEFAULT_RANDOMIZATION_WINDOW, epoch_size=INFINITELY_REPEAT, distributed_after=INFINITE_SAMPLES, multithreaded_deserializer=None):
        self['epochSize'] = cntk_py.SizeTWrapper(epoch_size) # force to store in size_t
        if not isinstance(deserializers, (list, tuple)):
            deserializers = [deserializers]
        self['deserializers'] = self.deserializers = deserializers or []
        self['randomize'] = randomize
        self['randomizationWindow'] = cntk_py.SizeTWrapper(randomization_window)
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
    def crop(crop_type='center', crop_size=0, side_ratio=0.0, area_ratio=0.0, aspect_ratio=1.0, jitter_type='none'):
        '''
        Crop transform that can be used to pass to `map_features`

        Args:
            crop_type (str, default 'center'): 'center', 'randomside', 'randomarea', 
             or 'multiview10'.  'randomside' and 'randomarea' are usually used during
             training, while 'center' and 'multiview10' are usually used during testing. 
             Random cropping is a popular data augmentation technique used to improve
             generalization of the DNN.
            crop_size (`int`, default 0): crop size in pixels. Ignored if set to 0. 
             When crop_size is non-zero, for example, crop_size=256, it means a cropping
             window of size 256x256 pixels will be taken. If one want to crop with
             non-square shapes, specify crop_size=256:224 will crop 256x224 (width x height) 
             pixels. `When crop_size is specified, side_ratio, area_ratio and aspect_ratio
             will be ignored.` 
            side_ratio (`float`, default 0.0): It specifies the ratio of final image 
             side (width or height) with respect to the original image. Ignored if set 
             to 0.0. Otherwise, must be set within `(0,1]`. For example, with an input 
             image size of 640x480, side_ratio of 0.5 means we crop a square region 
             (if aspect_ratio is 1.0) of the input image, whose width and height are 
             equal to 0.5*min(640, 480) = 240. To enable scale jitter (a popular data 
             augmentation technique), use colon-delimited values like side_ratio=0.5:0.75, 
             which means the crop will have size between 240 (0.5*min(640, 480)) and 360 
             (0.75*min(640, 480)). 
            area_ratio (`float`, default 0.0): It specifies the area ratio of final image 
             with respect to the original image. Ignored if set to 0.0. Otherwise, must be 
             set within `(0,1]`. For example, for an input image size of 200x150 pixels, 
             the area is 30,000. If area_ratio is 0.3333, we crop a square region (if 
             aspect_ratio is 1.0) with width and height equal to sqrt(30,000*0.3333)=100. 
             To enable scale jitter, use colon-delimited values such as area_ratio=0.3333:0.8, 
             which means the crop will have size between 100 (sqrt(30,000*0.3333)) and 
             155 (sqrt(30,000*0.8)). 
            aspect_ratio (`float`, default 1.0): It specifies the aspect ratio (width/height
             or height/width) of the crop window. Must be set within `(0,1]`. For example, 
             if due to size_ratio the crop size is 240x240, an aspect_ratio of 0.64 will 
             change the window size to non-square: 192x300 or 300x192, each having 50% 
             chance. Note the area of the crop window does not change. To enable aspect 
             ratio jitter, use colon-delimited values such as aspect_ratio=0.64:1.0, which means 
             the crop will have size between 192x300 (or euqally likely 300x192) and 240x240. 
            jitter_type (str, default 'none'): crop scale jitter type, possible
             values are 'none' and 'uniratio'. 'uniratio' means uniform distributed jitter
             scale between the minimum and maximum ratio values.

        Returns:
            dict describing the crop transform
        '''
        return dict(type='Crop', cropType=crop_type, cropSize=crop_size, sideRatio=side_ratio, 
                    areaRatio=area_ratio, aspectRatio=aspect_ratio, jitterType=jitter_type)

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

    @staticmethod
    def color(brightness_radius=0.0, contrast_radius=0.0, saturation_radius=0.0): 
        '''
        Color transform that can be used to pass to `map_features` for data augmentation.

        Args: 
            brightness_radius (float, default 0.0): Radius for brightness change. Must be 
             set within [0.0, 1.0]. For example, assume brightness_radius = 0.2, a random 
             number `x` is uniformly drawn from [-0.2, 0.2], and every pixel's value is 
             added by `x*meanVal`, where meanVal is the mean of the image pixel intensity 
             combining all color channels. 
            contrast_radius (float, default 0.0): Radius for contrast change. Must be 
             set within [0.0, 1.0]. For example, assume contrast_radius = 0.2, a random 
             number `x` is uniformly drawn from [-0.2, 0.2], and every pixel's value is 
             multiplied by `1+x`. 
            saturation_radius (float, default 0.0): Radius for saturation change. Only for
             color images and must be set within [0.0, 1.0]. For example, assume 
             saturation_radius = 0.2, a random number `x` is uniformly drawn from [-0.2, 0.2], 
             and every pixel's saturation is multiplied by `1+x`.

        Returns:
            dict describing the mean transform
        '''
        return dict(type='Color', brightnessRadius=brightness_radius, 
                    contrastRadius=contrast_radius, saturationRadius=saturation_radius)

    #@staticmethod
    #def intensity(intensity_stddev, intensity_file): 
    #    '''
    #    Intensity transform that can be used to pass to `map_features` for data augmentation. 
    #    Intensity jittering based on PCA transform as described in original `AlexNet paper
    #    <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    #    Currently uses precomputed values from 
    #    https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

    #    Args: 
    #        intensity_stddev (float): intensity standard deviation. 
    #        intensity_file (str): intensity file. 
    #    Returns:
    #        dict describing the mean transform        '''
    #    return dict(type='Intensity', intensityStdDev=intensity_stddev, intensityFile=intensity_file)

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


