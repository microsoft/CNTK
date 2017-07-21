# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import platform
import shelve
import shutil
import subprocess

import pytest


def test_deeprl():
    if platform.system() != 'Linux':
        pytest.skip('test only runs on Linux (Gym Atari dependency)')

    test_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(test_dir, '..', '..', '..', '..', 'Examples',
                              'ReinforcementLearning', 'deeprl', 'scripts')
    script_file = os.path.join(script_dir, 'run.py')
    config_file = os.path.join(script_dir, 'config_examples',
                               'qlearning.config')

    subprocess.call([
        'python', script_file, '--env=CartPole-v0', '--max_steps=6000',
        '--agent_config=' + config_file, '--eval_period=1000',
        '--eval_steps=20000'
    ])

    assert os.path.exists(
        os.path.join(test_dir, 'output', 'output.params')) == True

    wks = shelve.open(os.path.join(test_dir, 'output', 'output.wks'))
    rewards = wks['reward_history']
    assert len(rewards) >= 5 and len(rewards) <= 6
    assert max(rewards) >= 120

    shutil.rmtree(os.path.join(test_dir, 'output'))
