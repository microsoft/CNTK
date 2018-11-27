# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Factory method to create an agent."""

import configparser

from .policy_gradient import ActorCritic
from .qlearning import QLearning
from .random_agent import RandomAgent
from .tabular_qlearning import TabularQLearning


def make_agent(agent_config, o_space, a_space):
    """
    Choose appropriate method to create an agent.

    Args:
        agent_config: configure file specifying the agent type as well as
            training details.
        o_space: observation space, gym.spaces.tuple_space.Tuple is not
            supported.
        a_space: action space, limits to gym.spaces.discrete.Discrete.

    Returns:
        subclass inherited from :class:`.agent.AgentBaseClass`: QLearning,
            ActorCritic, TabularQLearning, or RandomAgent.
    """
    config = configparser.ConfigParser()
    config.read(agent_config)

    agent_type = config.get(
        'General', 'Agent', fallback='random').lower()
    agent = None
    if agent_type == 'qlearning':
        agent = QLearning(agent_config, o_space, a_space)
    elif agent_type == 'actor_critic':
        agent = ActorCritic(agent_config, o_space, a_space)
    elif agent_type == 'tabular_qlearning':
        agent = TabularQLearning(agent_config, o_space, a_space)
    elif agent_type == 'random':
        agent = RandomAgent(o_space, a_space)
    return agent
