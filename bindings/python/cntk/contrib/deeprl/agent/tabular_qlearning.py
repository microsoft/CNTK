# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Tabular Q-learning."""

import copy

import numpy as np

from .agent import AgentBaseClass
from .shared.qlearning_parameters import QLearningParameters


class TabularQLearning(AgentBaseClass):
    """Q-learning agent with tabular representation."""

    def __init__(self, cfg_filename, o_space, a_space):
        """Constructor for Q learning algorithm with tabular representation."""
        super(TabularQLearning, self).__init__(o_space, a_space)

        self._parameters = QLearningParameters(cfg_filename)
        if self._parameters.q_representation != 'tabular':
            raise ValueError(
                'Unexpected representation for tabular Q-learning: "{0}"'
                '\n'.format(self._parameters.q_representation))

        # Discretize the observation space if necessary
        if self._classname(o_space) != 'gym.spaces.discrete.Discrete':
            self._discretize_observation_space(
                o_space, self._parameters.discretization_resolution)

        self._q = self._parameters.initial_q + \
            np.zeros((self._num_states, self._num_actions))
        print('Initialized discrete Q-learning agent with {0} states and '
              '{1} actions.'.format(self._num_states, self._num_actions))

        self.episode_count = 0
        # step_count is incremented each time after receiving reward.
        self.step_count = 0

    def start(self, state):
        """Start a new episode."""
        self._adjust_exploration_rate()
        self._last_state = self._preprocess_state(state)
        self._last_action, action_behavior = \
            self._choose_action(self._last_state)
        self.episode_count += 1
        return self._last_action, {
            'action_behavior': action_behavior,
            'epsilon': self._epsilon}

    def step(self, reward, next_state):
        """Observe one transition and choose an action."""
        self._adjust_learning_rate()
        self.step_count += 1

        next_encoded_state = self._preprocess_state(next_state)
        td_err = reward + self._parameters.gamma * \
            np.max(self._q[next_encoded_state]) - \
            self._q[self._last_state, self._last_action]
        self._q[self._last_state, self._last_action] += self._eta * td_err

        self._adjust_exploration_rate()
        self._last_state = next_encoded_state
        self._last_action, action_behavior = self._choose_action(
            self._last_state)
        return self._last_action, {
            'action_behavior': action_behavior,
            'epsilon': self._epsilon}

    def end(self, reward, next_state):
        """Last observed reward/state of the episode (which then terminates)."""
        self._adjust_learning_rate()
        self.step_count += 1

        td_err = reward - self._q[self._last_state, self._last_action]
        self._q[self._last_state, self._last_action] += self._eta * td_err

    def set_as_best_model(self):
        """Copy current model to best model."""
        self._best_model = copy.deepcopy(self._q)

    def save(self, filename):
        """Save best model to file."""
        with open(filename, 'w') as f:
            for s in range(self._num_states):
                f.write('{0}\t{1}\n'.format(s, str(self._best_model[s])))

    def save_parameter_settings(self, filename):
        """Save parameter settings to file."""
        self._parameters.save(filename)

    def enter_evaluation(self):
        """Setup before evaluation."""
        self._epsilon = 0

    def _adjust_learning_rate(self):
        self._eta = self._parameters.eta_minimum + max(
            0,
            (self._parameters.initial_eta - self._parameters.eta_minimum) *
            (1 - float(self.step_count)/self._parameters.eta_decay_step_count))

    def _adjust_exploration_rate(self):
        self._epsilon = self._parameters.epsilon_minimum + max(
            0,
            (self._parameters.initial_epsilon - self._parameters.epsilon_minimum) *
            (1 - float(self.step_count)/self._parameters.epsilon_decay_step_count))

    def _choose_action(self, state):
        """Epsilon greedy policy."""
        if np.random.uniform(0, 1) < self._epsilon:
            return np.random.randint(self._num_actions), 'RANDOM'
        else:
            return np.argmax(self._q[state]), 'GREEDY'

    def _preprocess_state(self, state):
        """Discretize state to table row index."""
        o = self._discretize_state_if_necessary(state)
        return o
