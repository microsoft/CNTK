from .policy_gradient import ActorCritic
from .qlearning import QLearning
from .random_agent import RandomAgent
from .tabular_qlearning import TabularQLearning


def make_agent(agent_type, agent_config, o_space, a_space):
    """Choose appropriate method to create an agent."""
    normalized_agent_type = agent_type.lower()
    agent = None
    if normalized_agent_type == 'qlearning':
        agent = QLearning(agent_config, o_space, a_space)
    elif normalized_agent_type == 'actor_critic':
        agent = ActorCritic(agent_config, o_space, a_space)
    elif normalized_agent_type == 'tabular_qlearning':
        agent = TabularQLearning(agent_config, o_space, a_space)
    elif normalized_agent_type == 'random':
        agent = RandomAgent(o_space, a_space)
    return agent
