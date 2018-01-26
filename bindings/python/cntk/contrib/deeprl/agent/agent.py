# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Base class for defining an agent."""

from abc import ABCMeta, abstractmethod

import numpy as np

from importlib import import_module

from .shared.discretize import BoxSpaceDiscretizer


class AgentBaseClass(object):
    """Base class for defining an agent."""

    __metaclass__ = ABCMeta

    def __init__(self, o_space, a_space):
        """
        Constructor for AgentBaseClass.

        Args:
            o_space: observation space, gym.spaces.tuple_space.Tuple is not
                supported.
            a_space: action space, limits to gym.spaces.discrete.Discrete.
        """
        if self._classname(a_space) != 'gym.spaces.discrete.Discrete':
            raise ValueError(
                'Action space {0} incompatible with {1}. (Only supports '
                'Discrete action spaces.)'.format(a_space, self))
        self._num_actions = a_space.n

        # We assume the observation is in one of the following cases:
        # 1. discrete, and takes values from 0 to n - 1
        # 2. can be discretized, and the raw state is converted to an internal
        #    state taking values from 0 to n - 1
        # 3. raw, such as images from Atari games
        #
        # OpenAI gym supports the following observation types:
        # Discrete, Box, MultiBinary, MultiDiscrete and Tuple. Discrete
        # corresponds to case 1. Box, MultiBinary and MultiDiscrete can be
        # either case 2 or 3. Tuple is a mix of case 1, 2 or 3, and is not
        # supported currently.
        #
        # The observation-related parameters are defined as follows:
        # _discrete_observation_space: True for cases 1 and 2, False otherwise.
        #   State is represented by a scalar.
        # _space_discretizer: Not none for case 2 to indicate a conversion on
        #   state is required. None otherwise.
        # _shape_of_inputs: (n, ) for cases 1 and 2 to indicate it is a vector
        #   of length n. For case 3, it is the shape of array that represents
        #   the state. For example, an image input will have shape denoted as
        #   tuple (channel, width, height).
        if not (self._classname(o_space) == 'gym.spaces.discrete.Discrete' or
                self._classname(o_space) == 'gym.spaces.multi_binary.MultiBinary' or
                self._classname(o_space) == 'gym.spaces.box.Box' or
                self._classname(o_space) == 'gym.spaces.multi_discrete.MultiDiscrete'):
            raise ValueError(
                'Unsupported observation space type: {0}'.format(o_space))

        self._space_discretizer = None
        self._discrete_observation_space = \
            (self._classname(o_space) == 'gym.spaces.discrete.Discrete')
        # Set self._num_states for discrete observation space only.
        # Otherwise set it to None so that an exception will be raised
        # should it be used later in the code.
        self._num_states = \
            o_space.n if self._discrete_observation_space else None

        if (self._classname(o_space) == 'gym.spaces.discrete.Discrete' or
            self._classname(o_space) == 'gym.spaces.multi_binary.MultiBinary'):
            self._shape_of_inputs = (o_space.n,)
        else:
            self._shape_of_inputs = o_space.shape

        self._preprocessor = None
        self._best_model = None

    @abstractmethod
    def start(self, state):
        """
        Start a new episode.

        Args:
            state (object): observation provided by the environment.

        Returns:
            action (int): action choosen by agent.
            debug_info (dict): auxiliary diagnostic information.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def end(self, reward, next_state):
        """
        Last observed reward/state of the episode (which then terminates).

        Args:
            reward (float) : amount of reward returned after previous action.
            next_state (object): observation provided by the environment.
        """
        pass

    @abstractmethod
    def save(self, filename):
        """Save model to file."""
        pass

    @abstractmethod
    def save_parameter_settings(self, filename):
        """Save parameter settings to file."""
        pass

    @abstractmethod
    def set_as_best_model(self):
        """Copy current model to best model."""
        pass

    def enter_evaluation(self):
        """Setup before evaluation."""
        pass

    def exit_evaluation(self):
        """Tear-down after evaluation."""
        pass

    def evaluate(self, o):
        """
        Choose action for given observation without updating agent's status.

        Args:
            o (object): observation provided by the environment.

        Returns:
            action (int): action choosen by agent.
        """
        a, _ = self._choose_action(self._preprocess_state(o))
        return a

    @abstractmethod
    def _choose_action(self, state):
        """
        Choose an action according to the policy.

        Args:
            state (object): observation seen by agent, which can be different
                from what is provided by the environment. The difference comes
                from preprcessing.

        Returns:
            action (int): action choosen by agent.
            debug_info (str): auxiliary diagnostic information.
        """
        pass

    def _discretize_observation_space(self, space, discretization_resolution):
        if self._classname(space) == 'gym.spaces.box.Box':
            self._space_discretizer = BoxSpaceDiscretizer(
                space,
                discretization_resolution)
            self._discrete_observation_space = True
            self._num_states = self._space_discretizer.num_states
            self._shape_of_inputs = (self._num_states,)
        else:
            raise ValueError(
                "Unsupported space type for discretization: {0}".format(space))

    def _discretize_state_if_necessary(self, state):
        if self._space_discretizer is not None:
            return self._space_discretizer.discretize(state)
        else:
            return state

    def _index_to_vector(self, index, dimension):
        # TODO: consider using cntk.core.Value.one_hot here.
        a = np.zeros(dimension,)
        a[index] = 1
        return a

    def _preprocess_state(self, state):
        """Preprocess state to generate input to neural network.

        When state is a scalar which is the index of the state space, convert
        it using one-hot encoding.

        For other cases, state and input are the same, roughly.

        CNTK only supports float32 and float64. Performs appropriate
        type conversion as well.
        """
        o = self._discretize_state_if_necessary(state)
        if self._discrete_observation_space:
            o = self._index_to_vector(o, self._num_states)
        if self._preprocessor is not None:
            o = self._preprocessor.preprocess(o)
        # TODO: allow float64 dtype.
        if o.dtype.name != 'float32':
            o = o.astype(np.float32)
        return o

    def _classname(self, instance):
        return instance.__class__.__module__ + '.' + instance.__class__.__name__

    def _import_method(self, path):
        """Import method specified as module_name.method_name."""
        module_name, method_name = path.rsplit('.', 1)
        try:
            module = import_module(module_name)
            method = getattr(module, method_name)
        except (AttributeError, ImportError):
            raise ValueError('Cannot import method: "{0}"'.format(path))
        return method
