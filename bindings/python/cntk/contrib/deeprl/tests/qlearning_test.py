# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest
try:
    from unittest.mock import MagicMock, Mock, patch
except ImportError:
    # Note: separate install on Py 2.x (pip install mock)
    from mock import MagicMock, Mock, patch

import cntk.contrib.deeprl.tests.spaces as spaces
import numpy as np
from cntk.contrib.deeprl.agent.qlearning import QLearning
from cntk.contrib.deeprl.agent.shared.cntk_utils import huber_loss
from cntk.contrib.deeprl.agent.shared.replay_memory import _Transition
from cntk.layers import Dense
from cntk.losses import squared_error
from cntk.ops import input_variable


class QLearningTest(unittest.TestCase):
    """Unit tests for QLearning."""

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.Models.feedforward_network')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_dqn(self,
                      mock_parameters,
                      mock_model,
                      mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        mock_model.return_value = self._setup_test_model()

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)
        self.assertFalse(hasattr(sut, 'weight_variables'))
        self.assertIsNotNone(sut._trainer)
        mock_model.assert_called_with((1,), 2, '[2]', None)
        mock_replay_memory.assert_called_with(100, False)

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_dqn_prioritized_replay(self,
                                         mock_parameters,
                                         mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.use_prioritized_replay = True

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        self.assertIsNotNone(sut._weight_variables)
        mock_replay_memory.assert_called_with(100, True)

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_dqn_preprocessing(self,
                                    mock_parameters,
                                    mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.preprocessing = \
            'cntk.contrib.deeprl.agent.shared.preprocessing.AtariPreprocessing'
        mock_parameters.return_value.preprocessing_args = '()'

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        # Preprocessor with default arguments.
        self.assertIsNotNone(sut._preprocessor)
        self.assertEqual(sut._preprocessor.output_shape(), (4, 84, 84))

        # Preprocessor with arguments passed as a tuple.
        mock_parameters.return_value.preprocessing_args = '(3,)'
        sut = QLearning('', observation_space, action_space)
        self.assertEqual(sut._preprocessor.output_shape(), (3, 84, 84))

        # Preprocessor with inappropriate arguments.
        mock_parameters.return_value.preprocessing_args = '(3, 4)'
        self.assertRaises(
            TypeError, QLearning, '', observation_space, action_space)

        # Undefined preprocessor.
        mock_parameters.return_value.preprocessing = 'undefined'
        self.assertRaises(
            ValueError, QLearning, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.qlearning.Models.dueling_network')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_dueling_dqn(self, mock_parameters, mock_model):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.q_representation = 'dueling-dqn'
        mock_parameters.return_value.hidden_layers = '[2, [2], [2]]'
        mock_model.return_value = self._setup_test_model()

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)
        mock_model.assert_called_with((1,), 2, '[2, [2], [2]]', None)

    @patch('cntk.contrib.deeprl.agent.shared.customized_models.conv_dqn')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_customized_q(self, mock_parameters, mock_model):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.q_representation = \
            'cntk.contrib.deeprl.agent.shared.customized_models.conv_dqn'
        mock_model.return_value = self._setup_test_model()

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)
        mock_model.assert_called_with((1,), 2, None)

    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_unsupported_q(self, mock_parameters):
        instance = mock_parameters.return_value
        instance.q_representation = 'undefined'
        instance.preprocessing = ''

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        self.assertRaises(
            ValueError, QLearning, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.qlearning.Models.feedforward_network')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_init_dqn_huber_loss(self, mock_parameters, mock_model):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.use_error_clipping = True
        mock_model.return_value = self._setup_test_model()

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        mock_model.assert_called_with((1,), 2, '[2]', huber_loss)

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_update_q(self,
                      mock_parameters,
                      mock_replay_memory):
        """Test if _update_q_periodically() can finish successfully."""
        self._setup_parameters(mock_parameters.return_value)
        self._setup_replay_memory(mock_replay_memory.return_value)

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)
        sut._trainer.train_minibatch = MagicMock()
        sut._choose_action = MagicMock(side_effect=[
            (1, 'GREEDY'),
            (0, 'GREEDY'),
            (1, 'RANDOM'),
        ])

        action, debug_info = sut.start(np.array([0.1], np.float32))
        self.assertEqual(action, 1)
        self.assertEqual(debug_info['action_behavior'], 'GREEDY')
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 0)
        self.assertEqual(sut._epsilon, 0.1)
        self.assertEqual(sut._trainer.parameter_learners[0].learning_rate(), 0.1)
        self.assertEqual(sut._last_state, np.array([0.1], np.float32))
        self.assertEqual(sut._last_action, 1)

        action, debug_info = sut.step(1, np.array([0.2], np.float32))
        self.assertEqual(action, 0)
        self.assertEqual(debug_info['action_behavior'], 'GREEDY')
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 1)
        self.assertEqual(sut._epsilon, 0.09)
        # learning rate remains 0.1 as Q is not updated during this time step.
        self.assertEqual(sut._trainer.parameter_learners[0].learning_rate(), 0.1)
        self.assertEqual(sut._last_state, np.array([0.2], np.float32))
        self.assertEqual(sut._last_action, 0)

        action, debug_info = sut.step(2, np.array([0.3], np.float32))
        self.assertEqual(action, 1)
        self.assertEqual(debug_info['action_behavior'], 'RANDOM')
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 2)
        self.assertEqual(sut._epsilon, 0.08)
        self.assertEqual(sut._trainer.parameter_learners[0].learning_rate(), 0.08)
        self.assertEqual(sut._last_state, np.array([0.3], np.float32))
        self.assertEqual(sut._last_action, 1)

        sut.end(3, np.array([0.4], np.float32))
        self.assertEqual(sut.episode_count, 1)
        self.assertEqual(sut.step_count, 3)
        self.assertEqual(sut._epsilon, 0.08)
        # learning rate remains 0.08 as Q is not updated during this time step.
        self.assertEqual(sut._trainer.parameter_learners[0].learning_rate(), 0.08)

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_update_q_dqn(self,
                          mock_parameters,
                          mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        self._setup_replay_memory(mock_replay_memory.return_value)

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        sut._q.eval = \
            MagicMock(return_value=np.array([[[0.2, 0.1]]], np.float32))
        sut._target_q.eval = \
            MagicMock(return_value=np.array([[[0.3, 0.4]]], np.float32))
        sut._trainer = MagicMock()

        sut._update_q_periodically()

        np.testing.assert_array_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._input_variables],
            [np.array([0.1], np.float32)])
        # 10 (reward) + 0.9 (gamma) x 0.4 (max q_target) -> update action 0
        np.testing.assert_array_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._output_variables],
            [np.array([10.36, 0.1], np.float32)])

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_update_q_dqn_prioritized_replay(self,
                                             mock_parameters,
                                             mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.use_prioritized_replay = True
        self._setup_prioritized_replay_memory(mock_replay_memory.return_value)

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        def new_q_value(self):
            return np.array([[[0.2, 0.1]]], np.float32)
        sut._q.eval = MagicMock(side_effect=new_q_value)
        sut._target_q.eval = MagicMock(
            return_value=np.array([[[0.3, 0.4]]], np.float32))
        sut._trainer = MagicMock()

        sut._update_q_periodically()

        self.assertEqual(sut._trainer.train_minibatch.call_count, 1)
        np.testing.assert_array_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._input_variables],
            [
                np.array([0.1], np.float32),
                np.array([0.3], np.float32),
                np.array([0.1], np.float32)
            ])
        np.testing.assert_array_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._output_variables],
            [
                # 10 (reward) + 0.9 (gamma) x 0.4 (max q_target)
                np.array([10.36, 0.1], np.float32),
                # 11 (reward) + 0.9 (gamma) x 0.4 (max q_target)
                np.array([0.2, 11.36], np.float32),
                np.array([10.36, 0.1], np.float32)
            ])
        np.testing.assert_almost_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._weight_variables],
            [
                [0.16666667],
                [0.66666667],
                [0.16666667]
            ])
        self.assertAlmostEqual(
            sut._replay_memory.update_priority.call_args[0][0][3],
            105.2676)  # (10.16 + 0.1)^2
        self.assertAlmostEqual(
            sut._replay_memory.update_priority.call_args[0][0][4],
            129.0496,
            places=6)  # (11.26 + 0.1) ^ 2

    @patch('cntk.contrib.deeprl.agent.qlearning.ReplayMemory')
    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_update_q_double_dqn(self,
                                 mock_parameters,
                                 mock_replay_memory):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.double_q_learning = True
        self._setup_replay_memory(mock_replay_memory.return_value)

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        sut._q.eval = \
            MagicMock(return_value=np.array([[[0.2, 0.1]]], np.float32))
        sut._target_q.eval = \
            MagicMock(return_value=np.array([[[0.3, 0.4]]], np.float32))
        sut._trainer = MagicMock()

        sut._update_q_periodically()

        # 10 (reward) + 0.9 (gamma) x 0.3 -> update action 0
        np.testing.assert_array_equal(
            sut._trainer.train_minibatch.call_args[0][0][sut._output_variables],
            [np.array([10.27, 0.1], np.float32)])

    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_populate_replay_memory(self, mock_parameters):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.preprocessing = \
            'cntk.contrib.deeprl.agent.shared.preprocessing.SlidingWindow'
        mock_parameters.return_value.preprocessing_args = '(2, )'

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)

        sut._compute_priority = Mock(side_effect=[1, 2, 3])
        sut._choose_action = Mock(
            side_effect=[(0, ''), (0, ''), (1, ''), (1, '')])
        sut._replay_memory = MagicMock()
        sut._update_q_periodically = MagicMock()

        sut.start(np.array([0.1], np.float32))
        sut.step(0.1, np.array([0.2], np.float32))
        sut.step(0.2, np.array([0.3], np.float32))
        sut.end(0.3, np.array([0.4], np.float32))

        self.assertEqual(sut._replay_memory.store.call_count, 3)

        call_args = sut._replay_memory.store.call_args_list[0]
        np.testing.assert_array_equal(
            call_args[0][0],
            np.array([[0], [0.1]], np.float32))
        self.assertEqual(call_args[0][1], 0)
        self.assertEqual(call_args[0][2], 0.1)
        np.testing.assert_array_equal(
            call_args[0][3],
            np.array([[0.1], [0.2]], np.float32))
        self.assertEqual(call_args[0][4], 1)

        call_args = sut._replay_memory.store.call_args_list[2]
        np.testing.assert_array_equal(
            call_args[0][0],
            np.array([[0.2], [0.3]], np.float32))
        self.assertEqual(call_args[0][1], 1)
        self.assertEqual(call_args[0][2], 0.3)
        self.assertIsNone(call_args[0][3])
        self.assertEqual(call_args[0][4], 3)

    @patch('cntk.contrib.deeprl.agent.qlearning.QLearningParameters')
    def test_replay_start_size(self, mock_parameters):
        self._setup_parameters(mock_parameters.return_value)
        # Set exploration rate to 0
        mock_parameters.return_value.initial_epsilon = 0
        mock_parameters.return_value.epsilon_decay_step_count = 100
        mock_parameters.return_value.epsilon_minimum = 0
        mock_parameters.return_value.replay_start_size = 3

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = QLearning('', observation_space, action_space)
        sut._trainer = MagicMock()
        sut._replay_memory = MagicMock()

        _, debug = sut.start(np.array([0.1], np.float32))
        self.assertEqual(sut.step_count, 0)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)
        self.assertEqual(debug['action_behavior'], 'RANDOM')

        _, debug = sut.step(0.1, np.array([0.2], np.float32))
        self.assertEqual(sut.step_count, 1)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)
        self.assertEqual(debug['action_behavior'], 'RANDOM')

        sut.end(0.2, np.array([0.3], np.float32))
        self.assertEqual(sut.step_count, 2)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)

        _, debug = sut.start(np.array([0.4], np.float32))
        self.assertEqual(sut.step_count, 2)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)
        self.assertEqual(debug['action_behavior'], 'RANDOM')

        a, debug = sut.step(0.3, np.array([0.5], np.float32))
        self.assertEqual(sut.step_count, 3)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)
        self.assertEqual(debug['action_behavior'], 'GREEDY')

        a, debug = sut.start(np.array([0.6], np.float32))
        self.assertEqual(sut.step_count, 3)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 0)
        self.assertEqual(debug['action_behavior'], 'GREEDY')

        a, debug = sut.step(0.4, np.array([0.7], np.float32))
        self.assertEqual(sut.step_count, 4)
        self.assertEqual(sut._trainer.train_minibatch.call_count, 1)
        self.assertEqual(debug['action_behavior'], 'GREEDY')

    def _setup_parameters(self, parameters):
        parameters.q_representation = 'dqn'
        parameters.hidden_layers = '[2]'
        parameters.initial_epsilon = 0.1
        parameters.epsilon_decay_step_count = 9
        parameters.epsilon_minimum = 0.01
        parameters.initial_eta = 0.1
        parameters.eta_decay_step_count = 9
        parameters.eta_minimum = 0.01
        parameters.momentum = 0.95
        parameters.gradient_clipping_threshold = 10
        parameters.q_update_frequency = 2
        parameters.gamma = 0.9
        parameters.double_q_learning = False
        parameters.replay_start_size = 0
        parameters.replay_memory_capacity = 100
        parameters.use_prioritized_replay = False
        parameters.priority_alpha = 2
        parameters.priority_beta = 2
        parameters.priority_epsilon = 0.1
        parameters.preprocessing = ''
        parameters.use_error_clipping = False
        parameters.replays_per_update = 1

    def _setup_replay_memory(self, replay_memory):
        replay_memory.sample_minibatch.side_effect = \
            [[(0, _Transition(
                np.array([0.1], np.float32),
                0,
                10,
                np.array([0.2], np.float32),
                0.01))],
             [(1, _Transition(
                np.array([0.3], np.float32),
                1,
                -10,
                np.array([0.4], np.float32),
                0.02))]]

    def _setup_prioritized_replay_memory(self, replay_memory):
        # Duplicated values can be returned.
        replay_memory.sample_minibatch.return_value = \
            [(3, _Transition(
                np.array([0.1], np.float32),
                0,
                10,
                np.array([0.2], np.float32),
                2)),
             (4, _Transition(
                np.array([0.3], np.float32),
                1,
                11,
                np.array([0.4], np.float32),
                1)),
             (3, _Transition(
                np.array([0.1], np.float32),
                0,
                10,
                np.array([0.2], np.float32),
                2))]

    def _setup_test_model(self):
        inputs = input_variable(shape=(1,), dtype=np.float32)
        outputs = input_variable(shape=(1,), dtype=np.float32)

        q = Dense(1, activation=None)(inputs)
        loss = squared_error(q, outputs)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'f': q,
            'loss': loss
        }
