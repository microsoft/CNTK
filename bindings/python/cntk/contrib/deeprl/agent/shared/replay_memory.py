# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""Replay memory for Q learning."""

from __future__ import division

import math
import random
from collections import namedtuple

# Transition for experience replay.
#
# Args:
#   state: current state.
#   action: action applied to current state.
#   reward: scalar representing reward received by applying action to
#     current state.
#   next_state: the new state after action is applied.
#   priority: associated priority.
_Transition = namedtuple('Transition',
                         ['state', 'action', 'reward', 'next_state',
                          'priority'])


class ReplayMemory:
    """Replay memory to store samples of experience.

    Each transition is represented as (state, action, reward, next_state,
    priority) tuple. 'priority' is ignored for non-prioritized experience
    replay.
    """

    def __init__(self, capacity, prioritized=False):
        """Create replay memory with size capacity."""
        self._use_prioritized_replay = prioritized
        self._capacity = capacity
        # Position in the list where new experience will be written to.
        self._position = 0
        # For prioritized replay, 'sum-tree' data structure is used.
        # Transitions are stored in leaf nodes, while internal nodes store the
        # sum of priorities from all its descendants. List is used to represent
        # this complete binary tree. The following code initializes
        # all internal nodes, if any, to have value 0.
        self._memory = [0] * (capacity - 1) if prioritized else []

    def store(self, *args):
        """Store a transition in replay memory.

        If the memory is full, the oldest one gets overwritten.
        """
        if not self._isfull():
            self._memory.append(None)
        position = self._next_position_then_increment()
        old_priority = 0 if self._memory[position] is None \
            else self._memory[position].priority
        transition = _Transition(*args)
        self._memory[position] = transition
        if self._use_prioritized_replay:
            self._update_internal_nodes(
                position, transition.priority - old_priority)

    def update_priority(self, map_from_position_to_priority):
        """Update priority of transitions.

        Args:
            map_from_position_to_priority: dictionary mapping position of
                transition to its new priority. position should come from
                tuples returned by sample_minibatch().
        """
        if not self._use_prioritized_replay:
            return
        for position, new_priority in map_from_position_to_priority.items():
            old_priority = self._memory[position].priority
            self._memory[position] = _Transition(
                self._memory[position].state,
                self._memory[position].action,
                self._memory[position].reward,
                self._memory[position].next_state,
                new_priority)
            self._update_internal_nodes(
                position, new_priority - old_priority)

    def _actual_capacity(self):
        """Actual capacity needed.

        For prioritized replay, this includes both leaf nodes containing
        transitions and internal nodes containing priority sum.
        """
        return 2 * self._capacity - 1 \
            if self._use_prioritized_replay \
            else self._capacity

    def _isfull(self):
        return len(self._memory) == self._actual_capacity()

    def _next_position_then_increment(self):
        """Similar to position++."""
        start = self._capacity - 1 \
            if self._use_prioritized_replay \
            else 0
        position = start + self._position
        self._position = (self._position + 1) % self._capacity
        return position

    def _update_internal_nodes(self, index, delta):
        """Update internal priority sums when leaf priority has been changed.

        Args:
            index: leaf node index
            delta: change in priority
        """
        while index > 0:
            index = (index - 1) // 2
            self._memory[index] += delta

    def size(self):
        """Return the current number of transitions."""
        l = len(self._memory)
        if self._use_prioritized_replay:
            l -= (self._capacity - 1)
        return l

    def sample_minibatch(self, batch_size):
        """Sample minibatch of size batch_size."""
        pool_size = self.size()
        if pool_size == 0:
            return []

        if not self._use_prioritized_replay:
            chosen_idx = range(pool_size) \
                if pool_size <= batch_size \
                else random.sample(range(pool_size), batch_size)
        else:
            delta_p = self._memory[0] / batch_size
            chosen_idx = []
            for i in range(batch_size):
                lower = max(i * delta_p, 0)
                upper = min((i + 1) * delta_p, self._memory[0])
                p = random.uniform(lower, upper)
                chosen_idx.append(self._sample_with_priority(p))

        return [(i, self._memory[i]) for i in chosen_idx]

    def _sample_with_priority(self, p):
        parent = 0
        while True:
            left = 2 * parent + 1
            if left >= len(self._memory):
                # parent points to a leaf node already.
                return parent

            left_p = self._memory[left] if left < self._capacity - 1 \
                else self._memory[left].priority
            if p <= left_p:
                parent = left
            else:
                if left + 1 >= len(self._memory):
                    raise RuntimeError('Right child is expected to exist.')
                p -= left_p
                parent = left + 1
