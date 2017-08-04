# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Baseline agent that selects action uniformly randomly."""

import numpy as np

from .agent import AgentBaseClass


class RandomAgent(AgentBaseClass):
    """Agent that selects action uniformly randomly."""

    def __init__(self, o_space, a_space):
        """Constructor for RandomAgent."""
        super(RandomAgent, self).__init__(o_space, a_space)

        print('Initialized random agent with {0} actions.'.format(
            self._num_actions))

        self.episode_count = 0
        # step_count is incremented each time after receiving reward.
        self.step_count = 0

    def start(self, state):
        """Start a new episode."""
        self.episode_count += 1
        action, _ = self._choose_action(state)
        return action, {}

    def step(self, reward, next_state):
        """Observe one transition and choose an action."""
        self.step_count += 1
        action, _ = self._choose_action(next_state)
        return action, {}

    def end(self, reward, next_state):
        """Last observed reward/state of the episode (which then terminates)."""
        self.step_count += 1

    def set_as_best_model(self):
        """Copy current model to best model."""
        pass

    def save(self, filename):
        """Save best model to file."""
        pass

    def save_parameter_settings(self, filename):
        """Save parameter settings to file."""
        pass

    def _choose_action(self, state):
        """Random policy."""
        return np.random.randint(self._num_actions), 'RANDOM'
