from argparse import ArgumentParser

import gym
import numpy as np
import six
from cntk import Signature, TensorBoardProgressWriter
from cntk.core import Value
from cntk.device import set_default_device, cpu, gpu
from cntk.initializer import he_uniform
from cntk.layers import Convolution2D, Dense, default_options
from cntk.layers.typing import Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.models import Sequential
from cntk.ops import abs, element_select, less, relu, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.trainer import Trainer


class ReplayMemory(object):
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.uint8)

    def append(self, state, action, reward, done):
        """
        Appends the specified memory to the history.
        :param state: The state to append (should have the same shape as defined at initialization time)
        :param action: An integer representing the action done
        :param reward: An integer reprensenting the reward received for doing this action
        :param is_terminal: A boolean specifying if this state is a terminal (episode has finished)
        :return:
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """
        Generate a random minibatch. The returned indices can be retrieved using #get_state().
        See the method #minibatch() if you want to retrieve samples directly
        :param size: The minibatch size
        :param replace: Indicate if one index can appear multiple times (True), only once (False)
        :return: Indexes of the sampled states
        """

        # Local variable access are faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            # Check if replace=False to not include same index multiple times
            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """
        Generate a minibatch with the number of samples specified by the size parameter.
        :param size: Minibatch size
        :return: Tensor[minibatch_size, input_shape...)
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the visual memory
        :param index: State's index
        :return: Tensor[history_length, input_shape...]
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - self._history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)


class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation purpose
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        return self._buffer

    def append(self, state):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        self._buffer.fill(0)


class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy
    """

    def __init__(self, start, end, steps):
        self._steps = steps
        self._start = start
        self._stop = end

        self._a = -(end - start) / steps

    def __call__(self, nb_actions):
        return np.random.choice(nb_actions)

    def _epsilon(self, step):
        if step < 0:
            return self._stop
        elif step > self._steps:
            return self._start
        else:
            return self._a * step + self._stop

    def is_exploring(self, step):
        return np.random.rand() < self._epsilon(step)


class DeepQAgent(object):
    @staticmethod
    def huber_loss(y, y_hat, delta):
        """
        Compute the Huber Loss as part of the model graph
    
        Huber Loss is more robust to outliers. It is defined as:
         if |y - h_hat| < delta :
            0.5 * (y - y_hat)**2
        else :
            delta * |y - y_hat| - 0.5 * delta**2
    
        :param y: Target value
        :param y_hat: Estimated value
        :param delta: Outliers threshold
        :return: float
        """
        half_delta_squared = 0.5 * delta * delta
        error = y - y_hat
        abs_error = abs(error)

        less_than = 0.5 * square(error)
        more_than = (delta * abs_error) - half_delta_squared
        loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

        return reduce_sum(loss_per_sample, name='loss')

    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32, device_id=-1,
                 train_after=200000, train_interval=4, target_update_interval=10000):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(500000, input_shape, 4)
        self._action_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # CNTK Device setup
        set_default_device(cpu() if device_id == -1 else gpu(device_id))

        # Action Value model (used by agent to interact with the environment)
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Convolution2D((8, 8), 16, strides=4),
                Convolution2D((4, 4), 32, strides=2),
                Convolution2D((3, 3), 32, strides=1),
                Dense(256, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])

        # Define the loss, using Huber Loss (More robust to outliers)
        @Function
        @Signature(environment=Tensor[input_shape], actions=Tensor[nb_actions], q_targets=Tensor[1])
        def criterion(environment, actions, q_targets):
            # Define the loss, using Huber Loss (More robust to outliers)
            # actions is a sparse One Hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(environment) * actions, axis=0)

            # Define the trainer with Huber Loss function
            return DeepQAgent.huber_loss(q_targets, q_acted, 1.0)

        # Target model (used to compute target QValues in training process, updated less frequently)
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Define the learning rate (w.r.t to unit_gain=True)
        lr_per_sample = learning_rate / minibatch_size / (1 - momentum)
        lr_schedule = learning_rate_schedule(lr_per_sample, UnitType.sample)

        # Adam based SGD
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, unit_gain=True, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion)
        self._learner = l_sgd
        self._trainer = Trainer(self._action_value_net, (criterion, None), l_sgd, [self._metrics_writer])

    def act(self, state):
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._action_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                env_with_history.reshape((1,) + state.shape)  # Append batch axis with only one sample to evaluate
            )

            self._episode_q_means.append(np.mean(self._episode_q_means))
            self._episode_q_stddev.append(np.std(self._episode_q_stddev))

            action = q_values.argmax()

        # Keep track of interval action counter
        self._action_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        if (self._action_taken % self._train_interval) == 0:
            pre_states, actions, rewards, post_states, dones = self._memory.minibatch(self._minibatch_size)
            q_value_targets = self._compute_q(actions, rewards, post_states, dones)

            self._trainer.train_minibatch(
                self._action_value_net.argument_map(
                    environment=pre_states,
                    actions=Value.one_hot(actions, self.nb_actions),
                    q_targets=q_value_targets
                )
            )

        if (self._action_taken % self._target_update_interval) == 0:
            self._target_net = self._action_value_net.clone(CloneMethod.freeze)

    def _compute_q(self, actions, rewards, post_states, dones):
        q_hat = self._target_net.evaluate(post_states)
        q_hat_eval = q_hat.max(axis=1)
        q_targets = (1 - dones) * (self.gamma * q_hat_eval) + rewards

        return np.array(q_targets, dtype=np.float32)

    def _plot_metrics(self):
        """
        Plot current buffers accumulated values to visualize agent learning 
        :return: None
        """

        mean_q = 0 if len(self._episode_q_means) == 0 else np.asscalar(np.mean(self._episode_q_means))
        std_q = 0 if len(self._episode_q_stddev) == 0 else np.asscalar(np.mean(self._episode_q_stddev))
        sum_rewards = sum(self._episode_rewards)

        self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._action_taken)
        self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._action_taken)
        self._metrics_writer.write_value('Sum rewards per ep.', sum_rewards, self._action_taken)


def as_ale_input(environment):
    from PIL import Image
    return np.array(Image.fromarray(environment, 'RGB').convert('L').resize((84, 84)))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of epochs to run (epoch = 250k actions')
    parser.add_argument('-d', '--device', default=-1, type=int, help='-1 for CPU, >= 0 for GPU mapping GPU id')
    parser.add_argument('env', default='Pong-v3', type=str, metavar='N', nargs='?', help='Gym Atari environment to run')

    args = parser.parse_args()

    # 1. Make environment:
    env = gym.make(args.env)

    # 2. Make agent
    agent = DeepQAgent((4, 84, 84), env.action_space.n)

    # Train
    current_state = as_ale_input(env.reset())
    max_steps = args.epoch * 250000
    for step in six.moves.range(max_steps):
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
