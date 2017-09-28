# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Base class for defining preprocessing, as well as two concrete examples."""

from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
from PIL import Image


class Preprocessing(object):
    """Base class for defining preprocessing.

    All subclass constructors will take input_shape as the first argument.
    """

    __metaclass__ = ABCMeta

    def __init__(self, input_shape):
        """Constructor for base Preprocessing class."""
        self._input_shape = input_shape

    @abstractmethod
    def output_shape(self):
        """Return shape of preprocessed observation."""
        pass

    @abstractmethod
    def reset(self):
        """Reset preprocessing pipeline for new episode."""
        pass

    @abstractmethod
    def preprocess(self, observation):
        """Return preprocessed observation."""
        pass


class AtariPreprocessing(Preprocessing):
    """Preprocess screen images from Atari 2600 games.

    The image is represented by an array of shape (210, 160, 3). See
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    for more details.
    """

    def __init__(self, input_shape, history_len=4):
        super(AtariPreprocessing, self).__init__(input_shape)
        self.__history_len = history_len
        self.__processed_image_seq = deque(maxlen=history_len)
        self.reset()

    def output_shape(self):
        """Return shape of preprocessed Atari images."""
        return (self.__history_len, 84, 84)

    def reset(self):
        """Reset preprocessing pipeline for new episode."""
        self.__previous_raw_image = np.zeros(self._input_shape, dtype=np.uint8)
        self.__processed_image_seq.clear()
        for i in range(self.__history_len):
            self.__processed_image_seq.append(np.zeros((84, 84)))

    def preprocess(self, image):
        """Return preprocessed screen images from Atari 2600 games."""
        if image.shape != self._input_shape:
            raise ValueError(
                'Expecting image in shape {0} but get {1}\n'.format(
                    self._input_shape, image.shape))

        # Take the maximum value for each pixel over the current frame and the
        # previous one.
        im = Image.fromarray(
            np.maximum(image, self.__previous_raw_image), mode='RGB')

        # Extract luminance band.
        im = im.convert('YCbCr').split()[0]

        # Scale to 84 x 84
        im = im.resize((84, 84), Image.BILINEAR)

        self.__processed_image_seq.append(np.array(im))
        self.__previous_raw_image = image

        return np.stack(list(self.__processed_image_seq))


class SlidingWindow(Preprocessing):
    """Stack windowed inputs (x(t-m+1), ... x(t))."""

    def __init__(self, input_shape, history_len=4, dtype=np.float32):
        super(SlidingWindow, self).__init__(input_shape)
        self.__dtype = dtype
        self.__history_len = history_len
        self.__history = deque(maxlen=history_len)
        self.reset()

    def output_shape(self):
        """Return shape of preprocessed input."""
        return (self.__history_len,) + self._input_shape

    def reset(self):
        """Reset preprocessing pipeline for new episode."""
        self.__history.clear()
        for i in range(self.__history_len):
            self.__history.append(np.zeros(self._input_shape, self.__dtype))

    def preprocess(self, x):
        """Return preprocessed input x."""
        if x.shape != self._input_shape:
            raise ValueError(
                'Expecting input in shape {0} but get {1}\n'.format(
                    self._input_shape, x.shape))

        if x.dtype != self.__dtype:
            raise ValueError(
                'Expecting input in dtype {0} but get {1}\n'.format(
                    self.__dtype, x.dtype))

        self.__history.append(x)
        return np.stack(list(self.__history))
