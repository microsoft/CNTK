CNTK DeepRL toolkit implements deep Q learning (and its variants) and actor-critic method.
Tabular Q learning and random agent are also provided for baseline comparison.

The observation space and action space are represented by an OpenAI gym space type, see
https://github.com/openai/gym/tree/master/gym/spaces. Currently the toolkit limits
action space to be discrete https://github.com/openai/gym/blob/master/gym/spaces/discrete.py,
i.e., action is denoted by an integer between 0 and n-1 for n possible actions.
The observation space can be arbitrary expect Tuple https://github.com/openai/gym/blob/master/gym/spaces/tuple_space.py.

An example script is provided at CNTK/Examples/ReinforcementLearning/deeprl/scripts/run.py,
which interacts with environment, and does training and evaluation. Training details
are specified via a configure file. See CNTK/Examples/ReinforcementLearning/deeprl/config_examples
for example configure file for deep Q learning and actor-critic method.

For problem to be solved by deep RL algorithms, describe the problem as an environment following
the examples at CNTK/Examples/ReinforcementLearning/deeprl/env.

References:
deep Q learning
- DQN https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
- Prioritized Experience Replay https://arxiv.org/pdf/1511.05952.pdf
- Dueling Network https://arxiv.org/pdf/1511.06581.pdf
- Double Q Learning https://arxiv.org/pdf/1509.06461.pdf

actor-critic
- Actor-Critic https://arxiv.org/pdf/1602.01783.pdf
