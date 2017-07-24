# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class Maze2D(gym.Env):
    """This class creates a maze problem given a map."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self._load_map()
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.room_lengths[0] *
                                                 self.room_lengths[1])
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action, type(action))

        if (np.random.uniform(0, 1) > self.motion_noise):
            state0 = self.state[0]
            state1 = self.state[1]
            if action == 0:  # north
                state1 = np.minimum(self.room_lengths[1] - 1, state1 + 1)
            elif action == 1:  # east
                state0 = np.minimum(self.room_lengths[0] - 1, state0 + 1)
            elif action == 2:  # south
                state1 = np.maximum(0, state1 - 1)
            else:  # west
                state0 = np.maximum(0, state0 - 1)
            if not ([state0, state1] in self.wall_states):
                self.state[0] = state0
                self.state[1] = state1

        done = self._is_goal(self.state)
        reward = -1.0
        return self._encode_state(self.state), reward, done, {}

    def _reset(self):
        rnd_index = np.random.randint(0, len(self.initial_states))
        self.state = self.initial_states[rnd_index][:]
        return self._encode_state(self.state)

    def _load_map(self):
        self.room_lengths = np.array([25, 25])
        self.initial_states = [[0, 0]]
        self.goal_states = [[24, 24]]
        self.wall_states = []
        self._build_wall([2, 0], [2, 15])
        self._build_wall([5, 10], [5, 20])
        self._build_wall([5, 12], [13, 12])
        self._build_wall([15, 5], [15, 24])
        self._build_wall([10, 5], [22, 5])
        self.num_states = self.room_lengths[0] * self.room_lengths[1]
        self.motion_noise = 0.05

    def _is_goal(self, state):
        return self.state in self.goal_states

    def _encode_state(self, state):
        return int(state[1] * self.room_lengths[0] + state[0])

    def _build_wall(self, start, end):
        x_min = np.maximum(0, np.minimum(start[0], end[0]))
        x_max = np.minimum(self.room_lengths[0] - 1,
                           np.maximum(start[0], end[0]))
        y_min = np.maximum(0, np.minimum(start[1], end[1]))
        y_max = np.minimum(self.room_lengths[1] - 1,
                           np.maximum(start[1], end[1]))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if not ([x, y] in self.goal_states or
                        [x, y] in self.initial_states):
                    self.wall_states.append([x, y])

    def _render(self, mode='human', close=False):
        pass
