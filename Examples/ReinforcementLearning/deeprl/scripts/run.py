#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import argparse
import os
import shelve
import sys
import time
from contextlib import closing

import numpy as np
from gym import envs
from gym.envs.atari.atari_env import AtariEnv

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cntk.contrib.deeprl.agent import agent_factory
from env import env_factory


def new_episode():
    """Start a new episode.

    For Atari games, perform no-op actions at the beginning of the episode.
    """
    observation = env.reset()
    if args.render:
        env.render()
    if isinstance(env.env, AtariEnv):
        for t in range(args.num_noop):
            observation, reward, isTerminal, _ = env.step(0)
            if isTerminal:
                print('WARNING: Terminal signal received after {0} steps'
                      ''.format(t))
            if args.render:
                env.render()
    return observation


def evaluate_agent_if_necessary(eval_count, start_time):
    """Evaluate agent every --eval_period steps."""
    if agent.step_count >= eval_count * args.eval_period:
        elapsed_time = time.time() - start_time
        total_reward = 0
        num_episodes = 0
        episode_reward = 0
        i = 0
        agent.enter_evaluation()

        observation = new_episode()
        while i < args.eval_steps:
            i += 1
            action = agent.evaluate(observation)
            observation, reward, isTerminal, _ = env.step(action)
            if args.render:
                env.render()
            episode_reward += reward
            if isTerminal:
                num_episodes += 1
                total_reward += episode_reward
                episode_reward = 0
                observation = new_episode()

        reward = episode_reward if num_episodes == 0 \
            else total_reward / num_episodes
        print('\nAverage reward per episode after training {0} steps: {1}\n'
              ''.format(agent.step_count, reward))
        if len(reward_history) == 0 or reward > max(reward_history):
            agent.set_as_best_model()
        reward_history.append(reward)
        if len(training_time) != 0:
            elapsed_time += training_time[-1]
        training_time.append(elapsed_time)

        # Save results and update eval_count.
        filename_prefix = os.path.join(args.output_dir, args.output_dir)
        agent.save(filename_prefix + '.model')
        with closing(shelve.open(filename_prefix + '.wks',
                                 'n' if eval_count == 1 else 'c',
                                 0,
                                 True)) as shelf:
            if 'step_count' not in shelf:
                shelf['step_count'] = []
            shelf['step_count'].append(agent.step_count)
            shelf['reward_history'] = reward_history
            shelf['training_time_sec'] = training_time
        agent.exit_evaluation()
        eval_count += 1
        start_time = time.time()

    return eval_count, start_time


if __name__ == '__main__':
    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment that agent iteracts with.')
    parser.add_argument('--num_noop', type=int, default=30, help='Number of '
                        'no-op actions to be performed by the agent at the '
                        'start of an episode, for Atari environment only.')
    parser.add_argument('--agent_config', type=str, default='',
                        help='Config file for agent.')
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='Maximum steps to train an agent.')
    parser.add_argument('--max_episode_steps', type=int, default=0,
                        help='Maximum steps per episode. Use environment '
                        'specific value if 0.')
    parser.add_argument('--eval_period', type=int, default=250000,
                        help='Number of steps taken between each evaluation.')
    parser.add_argument('--eval_steps', type=int, default=125000,
                        help='Number of steps taken during each evaluation.')
    parser.add_argument('--verbose', action='store_true', help='Output debug '
                        'info if set to True.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory where workspace file and model file '
                        'are saved to. Model file will be named as '
                        'output_dir.model, and workspace file will be named '
                        'as output_dir.wks.')
    parser.add_argument('--render', action='store_true', help='Render '
                        'environment if set to True.')
    parser.add_argument('--seed', type=int, default=1234567, help='Seed for '
                        'random number generator. Negative value is ignored.')
    args = parser.parse_args()

    if (args.seed >= 0):
        np.random.seed(args.seed)

    # Use xrange for python 2.7 to speed up.
    if sys.version_info.major < 3:
        range = xrange

    # Create an OpenAI Gym environment, and obtain its state/action
    # information.
    if args.env not in envs.registry.env_specs.keys():
        # Try to find from local environment libraries.
        env_factory.register_env(args.env)
    env = envs.make(args.env)
    o_space = env.observation_space
    a_space = env.action_space
    image_observation = True if isinstance(
        env.env, AtariEnv) and env.env._obs_type == 'image' else False
    print("Loaded environment '{0}'".format(args.env))
    print("Observation space: '{0}'".format(o_space))
    print("Action space: '{0}'".format(a_space))
    print('Is observation an image: {0}'.format(image_observation))

    if args.max_episode_steps <= 0:
        args.max_episode_steps = \
            env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']

    # Create an agent.
    agent = agent_factory.make_agent(args.agent_config,
                                     o_space,
                                     a_space)

    # Create output folder, and save current parameter settings.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    agent.save_parameter_settings(
        os.path.join(args.output_dir, args.output_dir + '.params'))

    eval_count = 1
    reward_history = []
    training_time = []
    start_time = time.time()
    # Stop when maximum number of steps are reached.
    while agent.step_count < args.max_steps:
        # Evaluate agent every --eval_period steps.
        eval_count, start_time = evaluate_agent_if_necessary(
            eval_count, start_time)
        # Learn from new episode.
        observation = new_episode()
        action, debug_info = agent.start(observation)
        rewards = 0
        steps = 0
        for t in range(args.max_episode_steps):
            observation, reward, isTerminal, _ = env.step(action)
            if args.render:
                env.render()
            if args.verbose:
                print('\tStep\t{0}\t/\tAction\t{1},{2}\t/\tReward\t{3}'
                      ''.format(
                        agent.step_count,
                        action,
                        debug_info.get('action_behavior'),
                        reward))
            rewards += reward
            steps += 1
            if isTerminal:
                agent.end(reward, observation)
                break
            action, debug_info = agent.step(reward, observation)
        print('Episode {0}\t{1}/{2} steps\t{3} total reward\tterminated = {4}'
              ''.format(
                agent.episode_count, steps, agent.step_count, rewards, isTerminal))
        sys.stdout.flush()
    env.close()
