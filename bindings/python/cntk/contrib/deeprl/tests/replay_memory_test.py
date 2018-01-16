# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest

from cntk.contrib.deeprl.agent.shared.replay_memory import ReplayMemory


class ReplayMemoryTest(unittest.TestCase):
    """Unit tests for ReplayMemory."""

    def test_uniform_sampling(self):
        sut = ReplayMemory(3)
        self.assertEqual(sut.sample_minibatch(1), [])

        sut.store(1, 'ignore', 'ignore', 'ignore', 0)
        self.assertEqual(sut.size(), 1)
        self.assertEqual([s[0] for s in sut.sample_minibatch(1)], [0])
        self.assertEqual([s[0] for s in sut.sample_minibatch(2)], [0])

        sut.store(2, 'ignore', 'ignore', 'ignore', 0)
        sut.store(3, 'ignore', 'ignore', 'ignore', 0)
        self.assertEqual(sut.size(), 3)
        samples = sut.sample_minibatch(1)
        self.assertEqual(len(samples), 1)
        self.assertTrue(set(s[0] for s in samples).issubset([0, 1, 2]))
        self.assertTrue(set(s[1].state for s in samples).issubset([1, 2, 3]))

        sut.store(4, 'ignore', 'ignore', 'ignore', 0)
        self.assertEqual(sut.size(), 3)
        samples = sut.sample_minibatch(1)
        self.assertEqual(len(samples), 1)
        self.assertTrue(set(s[0] for s in samples).issubset([0, 1, 2]))
        self.assertTrue(set(s[1].state for s in samples).issubset([2, 3, 4]))

    def test_prioritized_sampling(self):
        sut = ReplayMemory(3, True)
        self.assertEqual(sut.sample_minibatch(1), [])

        sut.store(1, 'ignore', 'ignore', 'ignore', 1)
        self.assertEqual(sut.size(), 1)
        self.assertEqual([s[0] for s in sut.sample_minibatch(1)], [2])
        self.assertEqual([s[0] for s in sut.sample_minibatch(2)], [2, 2])

        sut.store(2, 'ignore', 'ignore', 'ignore', 3)
        sut.store(3, 'ignore', 'ignore', 'ignore', 2)
        self.assertEqual(sut.size(), 3)
        self.assertEqual(len(sut._memory), 5)
        self.assertEqual(sut._memory[:2], [6, 5])

        samples = sut.sample_minibatch(2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][0], 3)
        self.assertEqual(samples[0][1].state, 2)

        sut.store(4, 'ignore', 'ignore', 'ignore', 5)
        self.assertEqual(sut.size(), 3)
        self.assertEqual(sut._memory[:2], [10, 5])

        samples = sut.sample_minibatch(2)
        self.assertEqual(len(samples), 2)
        self.assertIn(samples[0][0], [3, 4])
        self.assertIn(samples[0][1].state, [2, 3])
        self.assertEqual(samples[1][0], 2)
        self.assertEqual(samples[1][1].state, 4)

        sut.update_priority({3: 4, 4: 0.5})
        self.assertEqual(sut._memory[:2], [9.5, 4.5])
