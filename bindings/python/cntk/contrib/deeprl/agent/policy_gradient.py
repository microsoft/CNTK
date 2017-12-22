# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Actor-Critic Policy Gradient."""

import cntk as C
import numpy as np

import ast

from .agent import AgentBaseClass
from .shared.cntk_utils import negative_of_entropy_with_softmax
from .shared.models import Models
from .shared.policy_gradient_parameters import PolicyGradientParameters


class ActorCritic(AgentBaseClass):
    """
    Actor-Critic Policy Gradient.

    See https://arxiv.org/pdf/1602.01783.pdf for a description of algorithm.
    """

    def __init__(self, config_filename, o_space, a_space):
        """
        Constructor for policy gradient.

        Args:
            config_filename: configure file specifying training details.
            o_space: observation space, gym.spaces.tuple_space.Tuple is not
                supported.
            a_space: action space, limits to gym.spaces.discrete.Discrete.
        """
        super(ActorCritic, self).__init__(o_space, a_space)

        self._parameters = PolicyGradientParameters(config_filename)

        # Create preprocessor.
        if self._parameters.preprocessing:
            preproc = self._import_method(self._parameters.preprocessing)
            self._preprocessor = preproc(
                self._shape_of_inputs,
                *ast.literal_eval(self._parameters.preprocessing_args))

        self._set_up_policy_network_and_value_network()

        self._trajectory_states = []
        self._trajectory_actions = []
        self._trajectory_rewards = []

        # Training data for the policy and value networks. Note they share the
        # same input.
        self._input_buffer = []
        self._value_network_output_buffer = []
        self._policy_network_output_buffer = []
        self._policy_network_weight_buffer = []

        self.episode_count = 0
        self.step_count = 0

    def start(self, state):
        """
        Start a new episode.

        Args:
            state (object): observation provided by the environment.

        Returns:
            action (int): action choosen by agent.
            debug_info (dict): auxiliary diagnostic information.
        """
        # Call _process_accumulated_trajectory() to process unused trajectory
        # data from previous episode.
        self._process_accumulated_trajectory(False)

        # Reset preprocessor.
        if self._preprocessor is not None:
            self._preprocessor.reset()

        # Append new state and action
        o = self._preprocess_state(state)
        action, _ = self._choose_action(o)
        self._trajectory_states.append(o)
        self._trajectory_actions.append(action)

        self.episode_count += 1

        return action, {}

    def step(self, reward, next_state):
        """
        Observe one transition and choose an action.

        Args:
            reward (float) : amount of reward returned after previous action.
            next_state (object): observation provided by the environment.

        Returns:
            action (int): action choosen by agent.
            debug_info (dict): auxiliary diagnostic information.
        """
        o = self._preprocess_state(next_state)
        self._trajectory_rewards.append(reward)
        self._trajectory_states.append(o)
        self.step_count += 1

        # Update every self._parameters.update_frequency
        if self.step_count % self._parameters.update_frequency == 0:
            self._process_accumulated_trajectory(True)
            self._update_networks()

        action, _ = self._choose_action(o)
        self._trajectory_actions.append(action)
        return action, {}

    def end(self, reward, next_state):
        """
        Last observed reward/state of the episode (which then terminates).

        Args:
            reward (float) : amount of reward returned after previous action.
            next_state (object): observation provided by the environment.
        """
        self._trajectory_rewards.append(reward)
        self.step_count += 1

        # Update every self._parameters.update_frequency
        if self.step_count % self._parameters.update_frequency == 0:
            self._process_accumulated_trajectory(False)
            self._update_networks()

    def set_as_best_model(self):
        """Copy current model to best model."""
        self._best_model = self._policy_network.clone('clone')

    def _set_up_policy_network_and_value_network(self):
        shape_of_inputs = self._shape_of_inputs if self._preprocessor is None \
            else self._preprocessor.output_shape()
        self._input_variables = \
            C.ops.input_variable(shape=shape_of_inputs, dtype=np.float32)

        # Set up policy network.
        if self._parameters.policy_representation == 'nn':
            model = Models.feedforward_network(
                shape_of_inputs,
                self._num_actions,
                self._parameters.policy_network_hidden_layers,
                C.losses.cross_entropy_with_softmax,
                use_placeholder_for_input=True)
        else:
            try:
                model_definition_function = self._import_method(
                    self._parameters.policy_representation)
                model = model_definition_function(
                    shape_of_inputs,
                    self._num_actions,
                    C.losses.cross_entropy_with_softmax,
                    use_placeholder_for_input=True)
            except ValueError:
                raise ValueError(
                    'Unknown representation for policy: "{0}"'
                    '\n'.format(self._parameters.policy_representation))

        self._policy_network = model['f']
        self._policy_network.replace_placeholder(self._input_variables)
        self._policy_network_output_variables = model['outputs']
        # The weight is computed as part of the Actor-Critic algorithm.
        self._policy_network_weight_variables = \
            C.ops.input_variable(shape=(1,), dtype=np.float32)
        self._policy_network_loss = \
            model['loss'] * self._policy_network_weight_variables

        # Initialized from a saved model.
        if self._parameters.initial_policy_network:
            self._policy_network.restore(
                self._parameters.initial_policy_network)

        print("Parameterized the agent's policy using neural networks "
              '"{0}" with {1} actions.\n'
              ''.format(self._parameters.policy_representation,
                        self._num_actions))

        # Set up value network.
        if self._parameters.shared_representation:
            # For shared representation, policy pi and value function V share
            # all non-output layers. To use cross_entropy_with_softmax loss
            # from cntk, _policy_network defined here doesn't include softmax
            # output layer. Therefore _value_network becomes _policy_network
            # plus one additional linear output layer.
            self._value_network = C.layers.Dense(1, activation=None)(
                self._policy_network)
            self._value_network_output_variables = C.ops.input_variable(
                shape=(1,), dtype=np.float32)
            self._value_network_loss = C.losses.squared_error(
                self._value_network, self._value_network_output_variables)
        else:
            if self._parameters.value_function_representation == 'nn':
                model = Models.feedforward_network(
                    shape_of_inputs,
                    1,  # value network outputs a scalar
                    self._parameters.value_network_hidden_layers,
                    use_placeholder_for_input=True)
            else:
                try:
                    model_definition_function = self._import_method(
                        self._parameters.value_function_representation)
                    model = model_definition_function(
                        shape_of_inputs,
                        1,  # value network outputs a scalar
                        use_placeholder_for_input=True)
                except ValueError:
                    raise ValueError(
                        'Unknown representation for value function: "{0}"'
                        '\n'.format(self._parameters.value_function_representation))

            self._value_network = model['f']
            self._value_network.replace_placeholder(self._input_variables)
            self._value_network_output_variables = model['outputs']
            self._value_network_loss = model['loss']  # squared_error by default

        combined_networks = C.ops.combine(
            [self._policy_network, self._value_network])
        combined_loss = self._policy_network_loss + \
            self._parameters.regularization_weight * \
            negative_of_entropy_with_softmax(self._policy_network) + \
            self._parameters.relative_step_size * self._value_network_loss

        # The learning rate will be updated later before each minibatch
        # training.
        # TODO: allow user to specify learner through config file.
        self._trainer = C.train.trainer.Trainer(
            combined_networks,
            (combined_loss, None),
            C.learners.adam(
                combined_networks.parameters,
                C.learners.learning_parameter_schedule_per_sample(
                    self._parameters.initial_eta),
                momentum=C.learners.momentum_schedule(self._parameters.momentum),
                variance_momentum=C.learners.momentum_schedule(0.999),
                minibatch_size=C.learners.IGNORE))

        print("Parameterized the agent's value function using neural network "
              '"{0}".\n'.format(
                self._parameters.policy_representation
                if self._parameters.shared_representation
                else self._parameters.value_function_representation))

    def _adjust_learning_rate(self):
        if self._parameters.initial_eta != self._parameters.eta_minimum:
            eta = self._parameters.eta_minimum + max(
                    0,
                    (self._parameters.initial_eta - self._parameters.eta_minimum) *
                    (1 - float(self.step_count)/self._parameters.eta_decay_step_count))
            self._trainer.parameter_learners[0].reset_learning_rate(
                C.learners.learning_parameter_schedule_per_sample(eta))

    def _choose_action(self, state):
        """
        Choose an action according to policy.

        Args:
            state (object): observation seen by agent, which can be different
                from what is provided by the environment. The difference comes
                from preprcessing.

        Returns:
            action (int): action choosen by agent.
            debug_info (object): probability vector the action is sampled from.
        """
        action_probs = \
            C.ops.softmax(self._evaluate_model(self._policy_network, state)).eval()
        return np.random.choice(self._num_actions, p=action_probs), action_probs

    def save(self, filename):
        """Save model to file."""
        self._best_model.save(filename)

    def save_parameter_settings(self, filename):
        """Save parameter settings to file."""
        self._parameters.save(filename)

    def _evaluate_model(self, model, state):
        r"""Evaluate log of pi(\cdot|state) or v(state)."""
        return np.squeeze(model.eval({model.arguments[0]: [state]}))

    def _process_accumulated_trajectory(self, keep_last):
        """Process accumulated trajectory to generate training data.

        Args:
            keep_last (bool): last state without action and reward will be kept
                if True.
        """
        if not self._trajectory_states:
            return

        # If trajectory hasn't terminated, we have _trajectory_states
        # and sometimes _trajectory_actions having one more item than
        # _trajectory_rewards. Same length is expected if called from
        # start() or end(), where the trajectory has terminiated.
        if len(self._trajectory_states) == len(self._trajectory_rewards):
            bootstrap_r = 0
        else:
            # Bootstrap from last state
            bootstrap_r = np.asscalar(self._evaluate_model(
                self._value_network, self._trajectory_states[-1]))
            last_state = self._trajectory_states.pop()
            if len(self._trajectory_actions) != len(self._trajectory_rewards):
                # This will only happen when agent calls start() to begin
                # a new episode without calling end() before to terminate the
                # prevous episode. The last action thus can be discarded.
                self._trajectory_actions.pop()

        if len(self._trajectory_states) != len(self._trajectory_rewards) or \
           len(self._trajectory_actions) != len(self._trajectory_rewards):
            raise RuntimeError("Can't pair (state, action, reward). "
                               "state/action can only be one more step ahead "
                               "of rewrad in trajectory.")

        for transition in zip(
                self._trajectory_states,
                self._trajectory_actions,
                self._discount_rewards(bootstrap_r)):
            self._input_buffer.append(transition[0])
            self._value_network_output_buffer.append([transition[2]])
            # TODO: consider using cntk.ops.one_hot instead of _index_to_vector
            self._policy_network_output_buffer.append(
                self._index_to_vector(transition[1], self._num_actions))
            self._policy_network_weight_buffer.append([transition[2]
                - self._evaluate_model(self._value_network, transition[0])])

        # Clear the trajectory history.
        self._trajectory_states = []
        self._trajectory_actions = []
        self._trajectory_rewards = []
        if keep_last:
            self._trajectory_states.append(last_state)

    def _update_networks(self):
        self._adjust_learning_rate()

        # Train the policy network on one minibatch.
        self._trainer.train_minibatch(
            {
                self._input_variables: np.array(self._input_buffer).astype(
                    np.float32),
                self._policy_network_output_variables:
                    np.array(self._policy_network_output_buffer).astype(
                        np.float32),
                self._policy_network_weight_variables:
                    np.array(self._policy_network_weight_buffer).astype(
                        np.float32),
                self._value_network_output_variables:
                    np.array(self._value_network_output_buffer).astype(
                        np.float32)
            })

        # Clear training data.
        self._input_buffer = []
        self._value_network_output_buffer = []
        self._policy_network_output_buffer = []
        self._policy_network_weight_buffer = []

    def _discount_rewards(self, bootstrap_r):
        discounted_rewards = [0] * len(self._trajectory_rewards)
        r = bootstrap_r
        for t in reversed(range(len(self._trajectory_rewards))):
            r = r * self._parameters.gamma + self._trajectory_rewards[t]
            discounted_rewards[t] = r
        return discounted_rewards
