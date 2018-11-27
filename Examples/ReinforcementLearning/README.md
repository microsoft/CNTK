# CNTK Examples: Reinforcement Learning

## Overview

|Data:     |Dynamic Atari Learning Environment (ALE)
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define Deep Q Neural Network (Mnih & al, 2013), a value-based Deep Reinforcement Learning method inspired by Q-Learning method 
|Network   |DeepQNeuralNetwork (DQN).
|Training  |Adam.
|Comments  |See below.

## Running the example

python DeepQNeuralNetwork.py

Some options are available: 

- -e : Number of epochs to run (one epoch is 250.000 actions taken)
- -p : Turn on tensorboard plotting, to visualize training
- Environment name, provided as trailing parameter to easily change the ALE environment
 
 Example:
 
 `
 python DeepQNeuralNetwork.py -e 200 -p SpaceInvaders-v3
 `

## Details

This example uses OpenAI Atari Learning Environment to train an agent on various Atari games using reinforcement learning.
As an Action-Value based network, a Deep Q Neural Network will try to estimate the expected reward for each action by looking
at the N last states (s_t, s_t-1, ..., s_t-N).

This agent has an exploration process, that allows it to take random actions to have a better understanding of the game dynamics.
The exploration process we use here is called 'Epsilon Greedy' where the 'best' action is taken with a probability of 1 - epsilon. 
Otherwise, a random action is taken. During the training, epsilon will slowly decay to a minimum value, commonly 0.1.

## Notes

This example **is only available on Linux** as OpenAI ALE doesn't provide Windows interface.
