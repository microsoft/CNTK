# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest
try:
    from unittest.mock import patch
except ImportError:
    # Note: separate install on Py 2.x (pip install mock)
    from mock import patch

import cntk.contrib.deeprl.tests.spaces as spaces
import numpy as np
from cntk.contrib.deeprl.agent.tabular_qlearning import TabularQLearning


class FakeTabularQLearning(TabularQLearning):
    """Override TabularQLearning for unittest."""

    def _choose_action(self, state):
        """Fake epsilon greedy policy."""
        return state % 2, 'GREEDY'


class TabularQLearningTest(unittest.TestCase):
    """Unit tests for TabularQLearning."""

    def test_init(self):
        # Discrete observation space.
        action_space = spaces.Discrete(2)
        observation_space = spaces.Discrete(3)
        sut = TabularQLearning('', observation_space, action_space)
        self.assertEqual(sut._num_actions, 2)
        self.assertEqual(sut._num_states, 3)
        self.assertEqual(sut._shape_of_inputs, (3, ))
        self.assertTrue(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)

        # Discretize observation space to default resolution.
        observation_space = spaces.Box(0, 1, (2,))
        sut = TabularQLearning('', observation_space, action_space)
        self.assertEqual(sut._num_states, 100)
        self.assertEqual(sut._shape_of_inputs, (100, ))
        self.assertTrue(sut._discrete_observation_space)
        self.assertIsNotNone(sut._space_discretizer)
        # Verify encoding of state
        self.assertEqual(sut._discretize_state_if_necessary([0, 0]), 0)
        self.assertEqual(sut._discretize_state_if_necessary([0.05, 0]), 0)
        self.assertEqual(sut._discretize_state_if_necessary([0.95, 0]), 90)
        self.assertEqual(sut._discretize_state_if_necessary([0, 0.05]), 0)
        self.assertEqual(sut._discretize_state_if_necessary([0, 0.95]), 9)
        self.assertEqual(sut._discretize_state_if_necessary([0.1, 0.2]), 12)
        self.assertEqual(sut._discretize_state_if_necessary([1, 1]), 99)

        # Unsupported observation space for tabular representation
        observation_space = spaces.MultiBinary(10)
        self.assertRaises(
            ValueError, TabularQLearning, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.tabular_qlearning.QLearningParameters')
    def test_init_unsupported_q(self, mock_qlearn_parameters):
        mock_qlearn_parameters.return_value.q_representation = 'undefined'

        action_space = spaces.Discrete(2)
        observation_space = spaces.Discrete(3)
        self.assertRaises(
            ValueError, TabularQLearning, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.tabular_qlearning.QLearningParameters')
    def test_update(self, mock_qlearn_parameters):
        self._setup_qlearn_parameters(mock_qlearn_parameters.return_value)
        action_space = spaces.Discrete(2)
        observation_space = spaces.Discrete(3)
        sut = FakeTabularQLearning('', observation_space, action_space)

        sut.start(0)
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 0)
        self.assertEqual(sut._epsilon, 0.1)
        # _eta has not been defined so far.
        self.assertEqual(sut._last_state, 0)
        self.assertEqual(sut._last_action, 0)

        sut.step(1, 1)
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 1)
        self.assertEqual(sut._epsilon, 0.09)
        self.assertEqual(sut._eta, 0.1)
        self.assertEqual(sut._last_state, 1)
        self.assertEqual(sut._last_action, 1)
        np.testing.assert_array_equal(
            sut._q, [[0.1, 0], [0, 0], [0, 0]])

        sut.step(1, 1)
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 2)
        self.assertEqual(sut._epsilon, 0.08)
        self.assertEqual(sut._eta, 0.09)
        self.assertEqual(sut._last_state, 1)
        self.assertEqual(sut._last_action, 1)
        np.testing.assert_array_equal(
            sut._q, [[0.1, 0], [0, 0.09], [0, 0]])

        sut.step(1, 1)
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 3)
        self.assertEqual(sut._epsilon, 0.07)
        self.assertEqual(sut._eta, 0.08)
        self.assertEqual(sut._last_state, 1)
        self.assertEqual(sut._last_action, 1)
        # 0.16928 = 0.09 + (1(reward) + 0.9(gamma)*max([0, 0.09]) - 0.09) * 0.08(eta)
        np.testing.assert_almost_equal(
            sut._q, [[0.1, 0], [0, 0.16928], [0, 0]])

        sut.end(1, 2)
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 4)
        # _epsilon remains the same as no action is chosen in end().
        self.assertEqual(sut._epsilon, 0.07)
        self.assertEqual(sut._eta, 0.07)
        # 0.2274304 = 0.16928 + (1(reward) - 0.16928) * 0.07(eta)
        np.testing.assert_almost_equal(
            sut._q, [[0.1, 0], [0, 0.2274304], [0, 0]])

    def _setup_qlearn_parameters(self, qlearn_parameters):
        qlearn_parameters.q_representation = 'tabular'
        qlearn_parameters.initial_q = 0
        qlearn_parameters.initial_epsilon = 0.1
        qlearn_parameters.epsilon_decay_step_count = 9
        qlearn_parameters.epsilon_minimum = 0.01
        qlearn_parameters.initial_eta = 0.1
        qlearn_parameters.eta_decay_step_count = 9
        qlearn_parameters.eta_minimum = 0.01
        qlearn_parameters.gamma = 0.9
