# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np


class Box:
    """Fake gym.spaces.box.Box to remove dependency on OpenAI gym."""

    def __init__(self, low, high, shape=None):
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

        self.__class__.__module__ = 'gym.spaces.box'

    @property
    def shape(self):
        return self.low.shape


class Discrete:
    """Fake gym.spaces.discrete.Discrete to remove dependency on OpenAI gym."""

    def __init__(self, n):
        self.n = n
        self.__class__.__module__ = 'gym.spaces.discrete'


class Tuple:
    """Fake gym.spaces.tuple_space.Tuple to remove dependency on OpenAI gym."""

    def __init__(self, spaces):
        self.spaces = spaces
        self.__class__.__module__ = 'gym.spaces.tuple_space'


class MultiBinary:
    """Fake gym.spaces.multi_binary.MultiBinary to remove dependency on OpenAI gym."""

    def __init__(self, n):
        self.n = n
        self.__class__.__module__ = 'gym.spaces.multi_binary'
