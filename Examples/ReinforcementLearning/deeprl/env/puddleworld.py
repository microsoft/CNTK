# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class PuddleWorld(gym.Env):
    """This class creates a continous-state maze problem given a map."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self._load_map()
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.zeros(2), self.room_lengths)
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action, type(action))

        if (np.random.uniform(0., 1.) > self.motion_noise):
            state0 = self.state[0]
            state1 = self.state[1]
            # Motion length is a truncated normal random variable.
            motion_length = np.maximum(
                0.,
                np.minimum(
                    self.motion_max,
                    np.random.normal(self.motion_mean, self.motion_std)))
            if action == 0:  # north
                state1 = np.minimum(self.room_lengths[1],
                                    state1 + motion_length)
            elif action == 1:  # east
                state0 = np.minimum(self.room_lengths[0],
                                    state0 + motion_length)
            elif action == 2:  # south
                state1 = np.maximum(0., state1 - motion_length)
            else:  # west
                state0 = np.maximum(0., state0 - motion_length)
            self.state[0] = state0
            self.state[1] = state1

        done = self._is_goal(self.state)
        reward = self._compute_reward(self.state)
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.copy(self.initial_state)
        return self.state

    def _load_map(self):
        self.room_lengths = np.array([1., 1.])
        self.initial_state = np.array([0., 0.])
        self.goal_state = np.array([1., 1.])
        self.goal_width = 0.01
        self.motion_noise = 0.05  # probability of no-motion (staying in same state)
        self.motion_mean = 0.1  # mean of motion length
        self.motion_std = 0.1 * self.motion_mean  # std of motion length
        self.motion_max = 2.0 * self.motion_mean
        self.puddle_centers = []
        self.puddle_radii = []
        self._build_puddle(np.array([0.2, 0.4]), 0.1)
        self._build_puddle(np.array([0.5, 0.8]), 0.1)
        self._build_puddle(np.array([0.9, 0.1]), 0.1)
        self.num_puddles = len(self.puddle_centers)
        self.puddle_cost = 2.0

    def _compute_reward(self, state):
        reward = -1
        for i in range(self.num_puddles):
            delta = state - self.puddle_centers[i]
            dist = np.dot(delta, delta)
            if dist <= self.puddle_radii[i]:
                reward -= self.puddle_cost
        return reward

    def _is_goal(self, state):
        return state[0] >= self.goal_state[0] - self.goal_width and \
            state[1] >= self.goal_state[1] - self.goal_width

    def _build_puddle(self, center, radius):
        self.puddle_centers.append(center)
        self.puddle_radii.append(radius)

    def _render(self, mode='human', close=False):
        pass
