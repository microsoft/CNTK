# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Discretize continuous environment space."""

import numpy as np


class BoxSpaceDiscretizer:
    """Discretize Box space."""

    def __init__(self, space, resolution):
        spaceclassname = \
            space.__class__.__module__ + '.' + space.__class__.__name__
        if spaceclassname != 'gym.spaces.box.Box':
            raise ValueError(
                'Space {0} incompatible with {1}. (Only supports '
                'Box space)'.format(space, self))

        assert np.isscalar(resolution) or space.low.shape == resolution.shape

        self._state_mins = space.low
        self._state_maxs = space.high
        if np.isscalar(resolution):
            self._state_resolutions = resolution + np.zeros(space.low.shape)
        else:
            self._state_resolutions = resolution
        self.num_states = int(np.prod(self._state_resolutions))

    def discretize(self, value):
        """Discretize box space observation."""
        index = 0
        for i, v in np.ndenumerate(value):
            i_idx = self._get_index(
                v,
                self._state_mins[i],
                self._state_maxs[i],
                self._state_resolutions[i])
            index = index * self._state_resolutions[i] + i_idx
        return int(index)

    def _get_index(self, value, minv, maxv, res):
        """Convert a continuous value to a discrete number."""
        if value >= maxv:
            return res - 1
        elif value <= minv:
            return 0
        else:
            ind = np.floor((value - minv) * res / (maxv - minv))
            return int(min(res - 1, max(0, ind)))
