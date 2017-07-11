import configparser

from .policy_gradient import ActorCritic
from .qlearning import QLearning
from .random_agent import RandomAgent
from .tabular_qlearning import TabularQLearning


def make_agent(agent_config, o_space, a_space):
    """Choose appropriate method to create an agent."""
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
