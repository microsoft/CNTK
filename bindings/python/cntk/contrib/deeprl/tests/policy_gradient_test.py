# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest
try:
    import unittest.mock as mock
    from unittest.mock import MagicMock, Mock, patch
except ImportError:
    # Note: separate install on Py 2.x (pip install mock)
    import mock
    from mock import MagicMock, Mock, patch

import cntk.contrib.deeprl.tests.spaces as spaces
import numpy as np
from cntk.contrib.deeprl.agent.policy_gradient import ActorCritic
from cntk.layers import Dense
from cntk.losses import cross_entropy_with_softmax
from cntk.ops import input_variable, placeholder


class PolicyGradientTest(unittest.TestCase):
    """Unit tests for policy gradient."""

    @patch('cntk.contrib.deeprl.agent.policy_gradient.Models.feedforward_network')
    def test_init(self, mock_model):
        mock_model.side_effect = self._setup_test_model

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)
        self.assertEqual(mock_model.call_count, 2)
        mock_model.assert_has_calls(
            [
                mock.call((1,), 2, '[10]', cross_entropy_with_softmax,
                    use_placeholder_for_input=True),
                mock.call((1,), 1, '[10]', use_placeholder_for_input=True)
            ],
            any_order=True)

    @unittest.skip("Skip this as CNTK can't reset UID during test.")
    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_init_from_existing_model(self, mock_parameters):
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(
            np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
        mock_parameters.return_value.policy_representation = 'nn'
        mock_parameters.return_value.policy_network_hidden_layers = '[2]'
        mock_parameters.return_value.initial_policy_network = \
            'tests/data/initial_policy_network.dnn'
        mock_parameters.return_value.preprocessing = ''

        sut = ActorCritic('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 3)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (2,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)

        # Incompatible network structure.
        mock_parameters.return_value.policy_network_hidden_layers = '[]'
        self.assertRaises(
            Exception, ActorCritic, '', observation_space, action_space)

        # Incompatible action space.
        mock_parameters.return_value.policy_network_hidden_layers = '[2]'
        action_space = spaces.Discrete(2)
        self.assertRaises(
            ValueError, ActorCritic, '', observation_space, action_space)

        # Incompatible observation space.
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(
            np.array([-1.2, -0.07, -1.0]), np.array([0.6, 0.07, 1.0]))
        self.assertRaises(
            ValueError, ActorCritic, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.policy_gradient.Models.feedforward_network')
    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_init_preprocess(self, mock_parameters, mock_model):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.preprocessing = \
            'cntk.contrib.deeprl.agent.shared.preprocessing.SlidingWindow'
        mock_parameters.return_value.preprocessing_args = '(2, )'
        mock_model.side_effect = self._setup_test_model

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        self.assertIsNotNone(sut._preprocessor)
        self.assertEqual(sut._preprocessor.output_shape(), (2, 1))
        self.assertEqual(mock_model.call_count, 2)
        mock_model.assert_has_calls(
            [
                mock.call((2, 1), 2, '[2]', cross_entropy_with_softmax,
                    use_placeholder_for_input=True),
                mock.call((2, 1), 1, '[2]', use_placeholder_for_input=True)
            ],
            any_order=True)

    @patch('cntk.contrib.deeprl.agent.shared.customized_models.conv_dqn')
    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_init_customized_model(self, mock_parameters, mock_model):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.policy_representation = \
            'cntk.contrib.deeprl.agent.shared.customized_models.conv_dqn'
        mock_parameters.return_value.value_function_representation = \
            'cntk.contrib.deeprl.agent.shared.customized_models.conv_dqn'
        mock_model.side_effect = self._setup_test_model

        sut = ActorCritic('', observation_space, action_space)

        self.assertEqual(mock_model.call_count, 2)
        mock_model.assert_has_calls(
            [
                mock.call((1,), 2, cross_entropy_with_softmax,
                    use_placeholder_for_input=True),
                mock.call((1,), 1, use_placeholder_for_input=True)
            ],
            any_order=True)

    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_init_unsupported_model(self, mock_parameters):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        self._setup_parameters(mock_parameters.return_value)

        # Verify sut can be constructed.
        sut = ActorCritic('', observation_space, action_space)

        mock_parameters.return_value.policy_representation = 'undefined'
        self.assertRaises(
            ValueError, ActorCritic, '', observation_space, action_space)

        mock_parameters.return_value.policy_representation = 'nn'
        mock_parameters.return_value.value_function_representation = 'undefined'
        self.assertRaises(
            ValueError, ActorCritic, '', observation_space, action_space)

    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_init_shared_representation(self, mock_parameters):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.shared_representation = True

        sut = ActorCritic('', observation_space, action_space)

        self.assertEqual(sut._num_actions, 2)
        self.assertIsNone(sut._num_states)
        self.assertEqual(sut._shape_of_inputs, (1,))
        self.assertFalse(sut._discrete_observation_space)
        self.assertIsNone(sut._space_discretizer)
        self.assertIsNone(sut._preprocessor)

        self.assertTrue(
            set(sut._policy_network.parameters).issubset(
                set(sut._value_network.parameters)))
        diff = set(sut._value_network.parameters).difference(
            set(sut._policy_network.parameters))
        # one for W and one for b
        self.assertEqual(len(diff), 2)

        shapes = []
        for item in diff:
            shapes.append(item.shape)
        self.assertEqual(set(shapes), {(2, 1), (1,)})

    def test_rollout(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        sut._choose_action = Mock(side_effect=[(0, ''), (1, ''), (1, '')])

        sut.start(np.array([0.1], np.float32))
        sut.step(0.1, np.array([0.2], np.float32))
        sut.step(0.2, np.array([0.3], np.float32))

        self.assertEqual(sut._trajectory_rewards, [0.1, 0.2])
        self.assertEqual(sut._trajectory_actions, [0, 1, 1])
        self.assertEqual(sut._trajectory_states, [0.1, 0.2, 0.3])

        sut.end(0.3, np.array([0.4], np.float32))

        self.assertEqual(sut._trajectory_rewards, [0.1, 0.2, 0.3])
        self.assertEqual(sut._trajectory_actions, [0, 1, 1])
        self.assertEqual(sut._trajectory_states, [0.1, 0.2, 0.3])

    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_rollout_preprocess(self, mock_parameters):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.preprocessing = \
            'cntk.contrib.deeprl.agent.shared.preprocessing.SlidingWindow'
        mock_parameters.return_value.preprocessing_args = '(2, "float32")'

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        sut._choose_action = Mock(side_effect=[(0, ''), (1, ''), (1, '')])

        sut.start(np.array([0.1], np.float32))
        sut.step(0.1, np.array([0.2], np.float32))
        sut.step(0.2, np.array([0.3], np.float32))

        self.assertEqual(sut._trajectory_rewards, [0.1, 0.2])
        self.assertEqual(sut._trajectory_actions, [0, 1, 1])
        np.testing.assert_array_equal(
            sut._trajectory_states,
            [
                np.array([[0], [0.1]], np.float32),
                np.array([[0.1], [0.2]], np.float32),
                np.array([[0.2], [0.3]], np.float32)
            ])

        sut.end(0.3, np.array([0.4], np.float32))

        self.assertEqual(sut._trajectory_rewards, [0.1, 0.2, 0.3])
        self.assertEqual(sut._trajectory_actions, [0, 1, 1])
        np.testing.assert_array_equal(
            sut._trajectory_states,
            [
                np.array([[0], [0.1]], np.float32),
                np.array([[0.1], [0.2]], np.float32),
                np.array([[0.2], [0.3]], np.float32)
            ])

    @patch('cntk.contrib.deeprl.agent.policy_gradient.PolicyGradientParameters')
    def test_rollout_with_update(self, mock_parameters):
        self._setup_parameters(mock_parameters.return_value)
        mock_parameters.return_value.update_frequency = 2

        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)
        sut._update_networks = MagicMock()

        sut._choose_action = Mock(side_effect=[
            (0, ''), (1, ''), (1, ''), (0, ''), (1, ''), (0, '')])

        sut.start(np.array([0.1], np.float32))
        sut.step(0.1, np.array([0.2], np.float32))
        self.assertEqual(sut._trajectory_rewards, [0.1])
        self.assertEqual(sut._trajectory_actions, [0, 1])
        self.assertEqual(sut._trajectory_states, [0.1, 0.2])
        self.assertEqual(sut._update_networks.call_count, 0)

        sut.step(0.2, np.array([0.3], np.float32))
        self.assertEqual(sut._trajectory_rewards, [])
        self.assertEqual(sut._trajectory_actions, [1])
        self.assertEqual(sut._trajectory_states, [0.3])
        self.assertEqual(sut._update_networks.call_count, 1)

        sut.step(0.3, np.array([0.4], np.float32))
        self.assertEqual(sut._trajectory_rewards, [0.3])
        self.assertEqual(sut._trajectory_actions, [1, 0])
        self.assertEqual(sut._trajectory_states, [0.3, 0.4])
        self.assertEqual(sut._update_networks.call_count, 1)

        sut.start(np.array([0.5], np.float32))
        self.assertEqual(sut._trajectory_rewards, [])
        self.assertEqual(sut._trajectory_actions, [1])
        self.assertEqual(sut._trajectory_states, [0.5])
        self.assertEqual(sut._update_networks.call_count, 1)

        sut.step(0.4, np.array([0.6], np.float32))
        self.assertEqual(sut._trajectory_rewards, [])
        self.assertEqual(sut._trajectory_actions, [0])
        self.assertEqual(sut._trajectory_states, [0.6])
        self.assertEqual(sut._update_networks.call_count, 2)

        sut.end(0.5, np.array([0.7], np.float32))
        self.assertEqual(sut._trajectory_rewards, [0.5])
        self.assertEqual(sut._trajectory_actions, [0])
        self.assertEqual(sut._trajectory_states, [0.6])
        self.assertEqual(sut._update_networks.call_count, 2)

    def test_process_accumulated_trajectory(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        # Set up.
        self._setup_trajectory(sut)

        # Call test method.
        sut._process_accumulated_trajectory(False)

        # Verify results.
        self.assertEqual(len(sut._trajectory_rewards), 0)
        self.assertEqual(len(sut._trajectory_actions), 0)
        self.assertEqual(len(sut._trajectory_states), 0)

        np.testing.assert_array_equal(
            sut._input_buffer,
            [np.array([0.1], np.float32), np.array([0.2], np.float32)])
        # For unknown reason, got [2.9974999999999996] instead of [2.9975] for
        # the following testcase, therefore use assert_array_almost_equal.
        np.testing.assert_array_almost_equal(
            sut._value_network_output_buffer,
            [
                [2.9975],    # 3.05 * 0.95 + 0.1
                [3.05]       # 3 (initial_r) * 0.95 + 0.2
            ])
        np.testing.assert_array_equal(
            sut._policy_network_output_buffer,
            [
                np.array([1, 0], np.float32),
                np.array([0, 1], np.float32)
            ]
        )
        np.testing.assert_array_almost_equal(
            sut._policy_network_weight_buffer,
            [
                [0.9975],    # 2.9975 - 2
                [2.05]       # 3.05 - 1
            ])

    def test_process_accumulated_trajectory_keep_last(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        # Set up.
        self._setup_trajectory(sut)

        # Call test method.
        sut._process_accumulated_trajectory(True)

        # Verify results.
        self.assertEqual(len(sut._trajectory_rewards), 0)
        self.assertEqual(len(sut._trajectory_actions), 0)
        self.assertEqual(sut._trajectory_states, [np.array([0.3], np.float32)])

    def test_update_policy_and_value_function(self):
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(0, 1, (1,))
        sut = ActorCritic('', observation_space, action_space)

        # Set up.
        self._setup_trajectory(sut)
        sut._process_accumulated_trajectory(True)
        sut._trainer = MagicMock()
        sut._adjust_learning_rate = MagicMock()

        # Call test method.
        sut._update_networks()

        # Verify value network behavior.
        self.assertEqual(
            sut._trainer.train_minibatch.call_count, 1)
        call_args = sut._trainer.train_minibatch.call_args
        np.testing.assert_array_equal(
            call_args[0][0][sut._input_variables],
            [np.array([0.1], np.float32), np.array([0.2], np.float32)])
        np.testing.assert_array_almost_equal(
            call_args[0][0][sut._value_network_output_variables],
            [[2.9975], [3.05]])
        np.testing.assert_array_equal(
            call_args[0][0][sut._policy_network_output_variables],
            [np.array([1, 0], np.float32), np.array([0, 1], np.float32)])
        np.testing.assert_array_almost_equal(
            call_args[0][0][sut._policy_network_weight_variables],
            [[0.9975], [2.05]])

        # Verify data buffer size.
        self.assertEqual(len(sut._input_buffer), 0)

    def _setup_parameters(self, params):
        params.policy_representation = 'nn'
        params.policy_network_hidden_layers = '[2]'
        params.value_function_representation = 'nn'
        params.value_network_hidden_layers = '[2]'
        params.relative_step_size = 0.5
        params.regularization_weight = 0.001
        params.initial_eta = 0.1
        params.eta_decay_step_count = 10
        params.eta_minimum = 0.01
        params.gamma = 0.9
        params.preprocessing = ''
        params.preprocessing_args = '()'
        params.shared_representation = False
        params.update_frequency = 4
        params.initial_policy_network = ''
        params.momentum = 0.95

    def _setup_trajectory(self, sut):
        # Corresponds to the case where sut.end() is not called.
        sut._trajectory_rewards = [0.1, 0.2]
        sut._trajectory_actions = [0, 1]
        sut._trajectory_states = [
            np.array([0.1], np.float32),
            np.array([0.2], np.float32),
            np.array([0.3], np.float32)]
        sut._value_network.eval = MagicMock(side_effect=[
            np.array([[[3]]], np.float32),
            np.array([[[2]]], np.float32),
            np.array([[[1]]], np.float32)])

    def _setup_test_model(self, *args, **kwargs):
        inputs = placeholder(shape=(1,))
        outputs = input_variable(shape=(1,), dtype=np.float32)

        q = Dense(1, activation=None)(inputs)
        loss = cross_entropy_with_softmax(q, outputs)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'f': q,
            'loss': loss
        }
