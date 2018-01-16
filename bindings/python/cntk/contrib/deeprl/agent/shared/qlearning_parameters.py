# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Q learning parameters."""

import numpy as np

import ast
import configparser


class QLearningParameters:
    """Parameters used by Q learning algorithm."""

    def __init__(self, config_file):
        """Read parameter values from config_file.

        Use default value if the value is not present.
        """
        # TODO: validate parameter values.
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(config_file)

        # Discount factor
        self.gamma = self.config.getfloat(
            'General', 'Gamma', fallback=0.95)

        # Name of class that does preprocessing.
        self.preprocessing = self.config.get(
            'General', 'PreProcessing', fallback='')

        # Arguments (except the first argument input_shape) of preprocessing as
        # a tuple.
        self.preprocessing_args = self.config.get(
            'General', 'PreProcessingArgs', fallback='()')

        # Representation of Q function, taking value from {'tabular', 'nn'}.
        self.q_representation = self.config.get(
            'QLearningAlgo', 'QRepresentation', fallback='tabular')

        # Initial value of epsilon (exploration rate), used by epsilon-greedy
        # policy.
        self.initial_epsilon = self.config.getfloat(
            'QLearningAlgo', 'InitialEpsilon', fallback=0.1)

        # Number of steps before epsilon reaches minimum value.
        self.epsilon_decay_step_count = self.config.getint(
            'QLearningAlgo', 'EpsilonDecayStepCount', fallback=100000)

        # Minimum value of epsilon.
        self.epsilon_minimum = self.config.getfloat(
            'QLearningAlgo', 'EpsilonMinimum', fallback=0.01)

        # Initial value of eta, which is the learning rate for gradient
        # descent.
        self.initial_eta = self.config.getfloat(
            'Optimization', 'InitialEta', fallback=0.001)

        # Number of steps before eta reaches minimum value.
        self.eta_decay_step_count = self.config.getint(
            'Optimization', 'EtaDecayStepCount', fallback=100000)

        # Minimum value of eta. Since Adam is used as the optimizer, a good
        # starting point is to set EtaMinimum equal to InitialEta, which is
        # equivalent to using a constant learning rate.
        self.eta_minimum = self.config.getfloat(
            'Optimization', 'EtaMinimum', fallback=0.001)

        # Momentum used by RMSProp.
        self.momentum = self.config.getfloat(
            'Optimization', 'Momentum', fallback=0.95)

        # Initial value for table entries.
        # TODO(maoyi): allow DQN initialization through config file.
        self.initial_q = self.config.getfloat(
            'QLearningAlgo', 'InitialQ', fallback=0.0)

        # Number of partitions for discretizing the continuous space. Either a
        # scalar which is applied to all dimensions, or a list specifying
        # different value for different dimension.
        self.discretization_resolution = ast.literal_eval(self.config.get(
            'QLearningAlgo', 'DiscretizationResolution', fallback='10'))
        if isinstance(self.discretization_resolution, list):
            self.discretization_resolution = np.array(
                self.discretization_resolution)

        # Number of actions chosen between successive
        # target network updates.
        self.target_q_update_frequency = self.config.getint(
            'QLearningAlgo', 'TargetQUpdateFrequency', fallback=10000)

        # Sample size of each minibatch.
        self.minibatch_size = self.config.getint(
            'QLearningAlgo', 'MinibatchSize', fallback=32)

        # Number of replays per update.
        self.replays_per_update = self.config.getint(
            'QLearningAlgo', 'ReplaysPerUpdate', fallback=1)

        # Number of actions chosen between successive SGD updates of Q.
        self.q_update_frequency = self.config.getint(
            'QLearningAlgo', 'QUpdateFrequency', fallback=4)

        # Use Huber loss with \delta=1 when True. Otherwise, use least square
        # loss.
        self.use_error_clipping = self.config.getboolean(
            'QLearningAlgo', 'ErrorClipping', fallback=True)

        # Capacity of replay memory.
        self.replay_memory_capacity = self.config.getint(
            'ExperienceReplay', 'Capacity', fallback=100000)

        # A uniform random policy is run for this number of steps to populate
        # replay memory.
        self.replay_start_size = self.config.getint(
            'ExperienceReplay', 'StartSize', fallback=5000)

        # Use prioritized replay. Fall back to uniform sampling when False .
        self.use_prioritized_replay = self.config.getboolean(
            'ExperienceReplay', 'Prioritized', fallback=False)

        # Used by prioritized replay, to determine how much prioritization is
        # used, with 0 corresponding to uniform.
        self.priority_alpha = self.config.getfloat(
            'ExperienceReplay', 'PriorityAlpha', fallback=0.7)

        # Used by prioritized replay, to anneal the amount of importance
        # sampling correction.
        self.priority_beta = self.config.getfloat(
            'ExperienceReplay', 'PriorityBeta', fallback=0.5)

        # Used by prioritized replay, to prevent transitions not being visited
        # once their error is zero.
        self.priority_epsilon = self.config.getfloat(
            'ExperienceReplay', 'PriorityEpsilon', fallback=0.01)

        # Number of nodes in each hidden layer, starting after the input layer.
        self.hidden_layers = self.config.get(
            'NetworkModel', 'HiddenLayerNodes', fallback='[20]')

        # Maximum norm of gradient per sample. No gradient clipping if the
        # parameter is missing from the config file.
        self.gradient_clipping_threshold = self.config.getfloat(
            'Optimization', 'GradientClippingThreshold', fallback=np.inf)

        # Use Double Q-learning if true.
        self.double_q_learning = self.config.getboolean(
            'QLearningAlgo', 'DoubleQLearning', fallback=False)

    def save(self, config_file):
        with open(config_file, 'w') as c:
            self.config.write(c)
