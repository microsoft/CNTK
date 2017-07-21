# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Policy Gradient parameters."""

import configparser


class PolicyGradientParameters:
    """Parameters used by Policy Gradient algorithms."""

    def __init__(self, config_file):
        """Read parameter values from config_file.

        Use default value if the parameter is not present.
        """
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(config_file)

        # Discount factor.
        self.gamma = self.config.getfloat(
            'General', 'Gamma', fallback=0.95)

        # Name of class that does preprocessing.
        self.preprocessing = self.config.get(
            'General', 'PreProcessing', fallback='')

        # Arguments (except the first argument input_shape) of preprocessing as
        # a tuple.
        self.preprocessing_args = self.config.get(
            'General', 'PreProcessingArgs', fallback='()')

        # If true, policy pi and value function V share all non-output layers.
        # PolicyRepresentation (and/or PolicyNetworkHiddenLayers) define
        # structure for all non-output layers. Policy then has one softmax
        # output layer, and value function has one linear output layer. If
        # false, all non-output layers of policy are still specified by
        # PolicyRepresentation. This is equivalent to defining unnormalized log
        # of policy pi. The value function, however, is completely specified by
        # ValueFunctionRepresentation (and/or ValueNetworkHiddenLayers), which
        # outputs a scalar.
        self.shared_representation = self.config.getboolean(
            'PolicyGradient', 'SharedRepresentation', fallback=False)

        # Representation of policy.
        self.policy_representation = self.config.get(
            'PolicyGradient', 'PolicyRepresentation', fallback='nn')

        # Suppose gradient of policy network is g, gradient of value network
        # is gv, during each update, policy network is updated as
        # \theta <- \theta + \eta * g where \eta is learning rate, and
        # value network is updated as
        # \theta_v <- \theta_v + \eta * relative_step_size * gv. This allows
        # policy network and value network to be updated at different learning
        # rates. Alternatively, this can be viewed as relative weight between
        # policy loss and value function loss.
        self.relative_step_size = self.config.getfloat(
            'PolicyGradient', 'RelativeStepSize', fallback=0.5)

        # Weight of regularization term.
        self.regularization_weight = self.config.getfloat(
            'PolicyGradient', 'RegularizationWeight', fallback=0.001)

        # Number of nodes in each hidden layer of policy network.
        self.policy_network_hidden_layers = self.config.get(
            'NetworkModel', 'PolicyNetworkHiddenLayerNodes', fallback='[10]')

        # Representation of value function.
        self.value_function_representation = self.config.get(
            'PolicyGradient', 'ValueFunctionRepresentation', fallback='nn')

        # Number of nodes in each hidden layer of value network.
        self.value_network_hidden_layers = self.config.get(
            'NetworkModel', 'ValueNetworkHiddenLayerNodes', fallback='[10]')

        # Initial value of eta, which is the learning rate for gradient descent.
        self.initial_eta = self.config.getfloat(
            'Optimization', 'InitialEta', fallback=0.001)

        # Number of steps before eta reaches minimum value.
        self.eta_decay_step_count = self.config.getint(
            'Optimization', 'EtaDecayStepCount', fallback=100000)

        # Minimum value of eta. Since Adam is used as the optimizer, a good
        # starting point is to set EtaMinimum equal to InitialEta, which is
        # equivalent to using a constant global learning rate cap, while Adam
        # continuously adapts individual parameter learning rates.
        self.eta_minimum = self.config.getfloat(
            'Optimization', 'EtaMinimum', fallback=0.001)

        # Momentum used by Adam.
        self.momentum = self.config.getfloat(
            'Optimization', 'Momentum', fallback=0.95)

        # Update frequency for policy network and value network, in the number
        # of time steps.
        self.update_frequency = self.config.getint(
            'PolicyGradient', 'UpdateFrequency', fallback=64)

        # Name of a file containing model of the same structure as policy
        # network (unnormalized log of policy pi), where model is obtained
        # through other methods (e.g. supervised learning), and saved by
        # cntk.ops.functions.Function.save(). Random initialization is
        # performed if value is empty.
        self.initial_policy_network = self.config.get(
            'PolicyGradient', 'InitialPolicy', fallback='')

    def save(self, config_file):
        with open(config_file, 'w') as c:
            self.config.write(c)
