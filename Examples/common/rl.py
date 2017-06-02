# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""Utility classes and functions for reforcement learning."""

from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple
import random

class Environment(object):
    """Abstract class for defining environment in reinforcement learning."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_current_state(self):
        """Returns environment's current state represented as tensor."""
        pass

    @abstractmethod
    def apply_action(self, action):
        """Applies action to the environment and returns reward value."""
        pass

# Transition for experience replay.
#
# Args:
#   state: current state.
#   action: action applied to current state.
#   reward: scalar representing reward received by applying action to
#     current state.
#   next_state: the new state after action is applied.
_Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

class _ReplayMemory(object):
    """Replay memory to store samples of experience, represented as
     (state, action, reward, next state) tuple.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, *args):
        """Stores a transition using FIFO."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = _Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_minibatch(self, batch_size):
        """Samples minibatch of size batch_size."""
        return random.sample(self.memory, batch_size)

def q_learning(environment, model):
    #TODO(maoyi): add Q-learning algorithm
    """Given environment and parameterized action value function Q(s, a),
    estimates the optimal action value function by Q-learning algorithm.

    Args:
        environment: environment the agent interacts with, inheriting from
            Environment class.
        model: parametric form of action value function.
    """
    return
