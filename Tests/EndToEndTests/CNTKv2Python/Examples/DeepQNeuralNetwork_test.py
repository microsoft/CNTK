# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import platform
import pytest
import gym

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "ReinforcementLearning"))

if platform.system() != 'Linux':
    pytest.skip('test only run on Linux (Gym Atari dependency)')

# 1. Make environment:
ENV_NAME = 'Pong-v3'
env = gym.make(ENV_NAME)

# 2. Make agent
agent = DeepQAgent((4, 84, 84), env.action_space.n, train_after=100, monitor=False)

# Train
current_step = 0
max_steps = 1000
current_state = as_ale_input(env.reset())

while current_step < max_steps:
    action = agent.act(current_state)
    new_state, reward, done, _ = env.step(action)
    new_state = as_ale_input(new_state)

    # Clipping reward for training stability
    reward = np.clip(reward, -1, 1)

    agent.observe(current_state, action, reward, done)
    agent.train()

    current_state = new_state

    if done:
        current_state = as_ale_input(env.reset())

    current_step += 1


assert len(agent._memory) == 1000