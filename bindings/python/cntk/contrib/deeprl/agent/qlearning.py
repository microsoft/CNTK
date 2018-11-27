# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Deep Q-learning and its variants."""

import math

import cntk as C
import numpy as np

import ast

from .agent import AgentBaseClass
from .shared.cntk_utils import huber_loss
from .shared.models import Models
from .shared.qlearning_parameters import QLearningParameters
from .shared.replay_memory import ReplayMemory


class QLearning(AgentBaseClass):
    """
    Q-learning agent.

    Including:
    - DQN https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    - Prioritized Experience Replay https://arxiv.org/pdf/1511.05952.pdf
    - Dueling Network https://arxiv.org/pdf/1511.06581.pdf
    - Double Q Learning https://arxiv.org/pdf/1509.06461.pdf
    """

    def __init__(self, config_filename, o_space, a_space):
        """Constructor for Q learning algorithm.

        Widely known as DQN. Use either predefined neural network structure
        (see models.py) or customized network (see customized_models.py).

        Args:
            config_filename: configure file specifying training details.
            o_space: observation space, gym.spaces.tuple_space.Tuple is not
                supported.
            a_space: action space, limits to gym.spaces.discrete.Discrete.
        """
        super(QLearning, self).__init__(o_space, a_space)

        self._parameters = QLearningParameters(config_filename)

        # Create preprocessor.
        if self._parameters.preprocessing:
            try:
                preproc = self._import_method(self._parameters.preprocessing)
                self._preprocessor = preproc(
                    self._shape_of_inputs,
                    *ast.literal_eval(self._parameters.preprocessing_args))
            except ValueError:
                raise ValueError(
                    'Unknown preprocessing method: "{0}"'
                    '\n'.format(self._parameters.preprocessing))

        # Set up the Q-function.
        shape_of_inputs = self._shape_of_inputs \
            if self._preprocessor is None \
            else self._preprocessor.output_shape()
        if self._parameters.q_representation == 'dqn':
            model = Models.feedforward_network(
                shape_of_inputs,
                self._num_actions,
                self._parameters.hidden_layers,
                huber_loss if self._parameters.use_error_clipping else None)
        elif self._parameters.q_representation == 'dueling-dqn':
            model = Models.dueling_network(
                shape_of_inputs,
                self._num_actions,
                self._parameters.hidden_layers,
                huber_loss if self._parameters.use_error_clipping else None)
        else:
            try:
                model_definition_function = self._import_method(
                    self._parameters.q_representation)
                model = model_definition_function(
                    shape_of_inputs,
                    self._num_actions,
                    huber_loss if self._parameters.use_error_clipping else None)
            except ValueError:
                raise ValueError(
                    'Unknown representation for Q-learning: "{0}"'
                    '\n'.format(self._parameters.q_representation))

        self._q = model['f']
        self._input_variables = model['inputs']
        self._output_variables = model['outputs']
        if self._parameters.use_prioritized_replay:
            self._weight_variables = \
                C.ops.input_variable(shape=(1,), dtype=np.float32)
            self._loss = model['loss'] * self._weight_variables
        else:
            self._loss = model['loss']


        minibatch_size = int(self._parameters.minibatch_size)
        # If gradient_clipping_threshold_per_sample is inf, gradient clipping
        # will not be performed. Set gradient_clipping_with_truncation to False
        # to clip the norm.
        # TODO: allow user to specify learner through config file.
        opt = C.learners.adam(
            self._q.parameters,
            C.learners.learning_parameter_schedule_per_sample(
                self._parameters.initial_eta),
            use_mean_gradient=True,
            momentum=C.learners.momentum_schedule(self._parameters.momentum),
            variance_momentum=C.learners.momentum_schedule(0.999),
            gradient_clipping_threshold_per_sample=
                self._parameters.gradient_clipping_threshold,
            gradient_clipping_with_truncation=False)
        self._trainer = C.train.trainer.Trainer(
            self._q, (self._loss, None), opt)

        # Initialize target Q.
        self._target_q = self._q.clone('clone')

        # Initialize replay memory.
        self._replay_memory = ReplayMemory(
            self._parameters.replay_memory_capacity,
            self._parameters.use_prioritized_replay)

        print('Parameterized Q-learning agent using neural networks '
              '"{0}" with {1} actions.\n'
              ''.format(self._parameters.q_representation,
                        self._num_actions))

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
        if self._preprocessor is not None:
            self._preprocessor.reset()

        self._adjust_exploration_rate()
        self._last_state = self._preprocess_state(state)
        self._last_action, action_behavior = \
            self._choose_action(self._last_state)
        self.episode_count += 1
        return self._last_action, {
            'action_behavior': action_behavior,
            'epsilon': self._epsilon}

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
        next_encoded_state = self._preprocess_state(next_state)
        priority = self._compute_priority(
            self._last_state, self._last_action, reward, next_encoded_state)
        self._replay_memory.store(
            self._last_state,
            self._last_action,
            reward,
            next_encoded_state,
            priority)
        self.step_count += 1

        # Update Q every self._parameters.q_update_frequency
        self._update_q_periodically()

        self._adjust_exploration_rate()
        self._last_state = next_encoded_state
        self._last_action, action_behavior = self._choose_action(
            self._last_state)
        return self._last_action, {
            'action_behavior': action_behavior,
            'epsilon': self._epsilon}

    def end(self, reward, next_state):
        """
        Last observed reward/state of the episode (which then terminates).

        Args:
            reward (float) : amount of reward returned after previous action.
            next_state (object): observation provided by the environment.
        """
        priority = self._compute_priority(
            self._last_state, self._last_action, reward, None)
        self._replay_memory.store(
            self._last_state,
            self._last_action,
            reward,
            None,
            priority)
        self.step_count += 1

        # Update Q every self._parameters.q_update_frequency
        self._update_q_periodically()

    def set_as_best_model(self):
        """Copy current model to best model."""
        self._best_model = self._q.clone('clone')

    def enter_evaluation(self):
        """Setup before evaluation."""
        self._epsilon = 0

    def _adjust_learning_rate(self):
        if self._parameters.initial_eta != self._parameters.eta_minimum:
            eta = self._parameters.eta_minimum + max(
                0,
                (self._parameters.initial_eta - self._parameters.eta_minimum) *
                (1 - float(self.step_count)/self._parameters.eta_decay_step_count))

            self._trainer.parameter_learners[0].reset_learning_rate(
                C.learners.learning_parameter_schedule_per_sample(
                    eta))

    def _adjust_exploration_rate(self):
        self._epsilon = self._parameters.epsilon_minimum + max(
            0,
            (self._parameters.initial_epsilon - self._parameters.epsilon_minimum) *
            (1 - float(self.step_count)/self._parameters.epsilon_decay_step_count))

    def _choose_action(self, state):
        """
        Epsilon greedy policy.

        Args:
            state (object): observation seen by agent, which can be different
                from what is provided by the environment. The difference comes
                from preprcessing.

        Returns:
            action (int): action choosen by agent.
            debug_info (str): auxiliary diagnostic information.
        """
        if self.step_count < self._parameters.replay_start_size or \
                np.random.uniform(0, 1) < self._epsilon:
            return np.random.randint(self._num_actions), 'RANDOM'
        else:
            return np.argmax(self._evaluate_q(self._q, state)), 'GREEDY'

    def save(self, filename):
        """Save model to file."""
        self._best_model.save(filename)

    def save_parameter_settings(self, filename):
        """Save parameter settings to file."""
        self._parameters.save(filename)

    def _evaluate_q(self, model, state, action=None):
        """
        Evaluate Q[state, action].

        If action is None, return values for all actions.
        Args:
            state (object): observation seen by agent, which can be different
                from what is provided by the environment. The difference comes
                from preprcessing.
            action (int): action choosen by agent.
        """
        q = np.squeeze(model.eval({model.arguments[0]: [state]}))
        if action is None:
            return q
        else:
            return q[action]

    def _update_q_periodically(self):
        if self.step_count < self._parameters.replay_start_size or \
                self.step_count % self._parameters.q_update_frequency != 0:
            return

        self._adjust_learning_rate()
        for i in range(self._parameters.replays_per_update):
            self._replay_and_update()

        # Clone target network periodically.
        if self.step_count % \
                self._parameters.target_q_update_frequency == 0:
            self._target_q = self._q.clone('clone')

    def _replay_and_update(self):
        """Perform one minibatch update of Q."""
        input_values = []
        output_values = []
        if self._parameters.use_prioritized_replay:
            # importance sampling weights.
            weight_values = []

        minibatch = self._replay_memory.sample_minibatch(
            self._parameters.minibatch_size)
        for index_transition_pair in minibatch:
            input_value = index_transition_pair[1].state

            # output_value is the same for all actions except last_action.
            output_value = self._evaluate_q(
                self._q, index_transition_pair[1].state)
            td_err = self._compute_td_err(
                index_transition_pair[1].state,
                index_transition_pair[1].action,
                index_transition_pair[1].reward,
                index_transition_pair[1].next_state)
            output_value[index_transition_pair[1].action] += td_err

            input_values.append(input_value)
            output_values.append(output_value)

            if self._parameters.use_prioritized_replay:
                weight_values.append(math.pow(
                    index_transition_pair[1].priority,
                    -self._parameters.priority_beta))

        if self._parameters.use_prioritized_replay:
            w_sum = sum(weight_values)
            weight_values = [[w / w_sum] for w in weight_values]
            self._trainer.train_minibatch(
                {
                    self._input_variables: np.array(input_values).astype(
                        np.float32),
                    self._output_variables: np.array(output_values).astype(
                        np.float32),
                    self._weight_variables: np.array(weight_values).astype(
                        np.float32)
                })

            # Update replay priority.
            position_priority_map = {}
            for index_transition_pair in minibatch:
                position_priority_map[index_transition_pair[0]] = \
                    self._compute_priority(
                        index_transition_pair[1].state,
                        index_transition_pair[1].action,
                        index_transition_pair[1].reward,
                        index_transition_pair[1].next_state)

            self._replay_memory.update_priority(position_priority_map)
        else:
            self._trainer.train_minibatch(
                {
                    self._input_variables: np.array(input_values).astype(
                        np.float32),
                    self._output_variables: np.array(output_values).astype(
                        np.float32)
                })

    def _compute_td_err(self, state, action, reward, next_state):
        td_err = reward
        if next_state is not None:
            if self._parameters.double_q_learning:
                td_err += self._parameters.gamma * \
                    self._evaluate_q(
                        self._target_q,
                        next_state,
                        np.argmax(self._evaluate_q(self._q, next_state)))
            else:
                td_err += self._parameters.gamma * np.max(
                    self._evaluate_q(self._target_q, next_state))
        td_err -= self._evaluate_q(self._q, state, action)
        return td_err

    def _compute_priority(self, state, action, reward, next_state):
        priority = None
        if self._parameters.use_prioritized_replay:
            priority = math.pow(
                math.fabs(self._compute_td_err(
                    state, action, reward, next_state))
                + self._parameters.priority_epsilon,
                self._parameters.priority_alpha)
        return priority
