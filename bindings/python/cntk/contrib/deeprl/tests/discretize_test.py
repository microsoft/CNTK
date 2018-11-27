# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest

import cntk.contrib.deeprl.tests.spaces as spaces
import numpy as np
from cntk.contrib.deeprl.agent.shared.discretize import BoxSpaceDiscretizer


class BoxSpaceDiscretizerTest(unittest.TestCase):
    """Unit tests for BoxSpaceDiscretizer."""

    def test_scalar(self):
        s = spaces.Box(0, 1, (2,))
        sut = BoxSpaceDiscretizer(s, 10)

        self.assertEqual(sut.discretize([0, 0]), 0)
        self.assertEqual(sut.discretize([0.05, 0]), 0)
        self.assertEqual(sut.discretize([0.95, 0]), 90)
        self.assertEqual(sut.discretize([0, 0.05]), 0)
        self.assertEqual(sut.discretize([0, 0.95]), 9)
        self.assertEqual(sut.discretize([0.1, 0.2]), 12)
        self.assertEqual(sut.discretize([1, 1]), 99)

    def test_list(self):
        s = spaces.Box(0, 1, (2,))
        sut = BoxSpaceDiscretizer(s, np.array([10, 2]))

        self.assertEqual(sut.discretize([0, 0]), 0)
        self.assertEqual(sut.discretize([0.05, 0]), 0)
        self.assertEqual(sut.discretize([0.95, 0]), 18)
        self.assertEqual(sut.discretize([0, 0.05]), 0)
        self.assertEqual(sut.discretize([0, 0.95]), 1)
        self.assertEqual(sut.discretize([0.1, 0.2]), 2)
        self.assertEqual(sut.discretize([1, 1]), 19)

        sut = BoxSpaceDiscretizer(s, np.array([10, 1]))

        self.assertEqual(sut.discretize([0, 0]), 0)
        self.assertEqual(sut.discretize([0.05, 0]), 0)
        self.assertEqual(sut.discretize([0.95, 0]), 9)
        self.assertEqual(sut.discretize([0, 0.05]), 0)
        self.assertEqual(sut.discretize([0, 0.95]), 0)
        self.assertEqual(sut.discretize([0.1, 0.2]), 1)
        self.assertEqual(sut.discretize([1, 1]), 9)

    def test_array(self):
        s = spaces.Box(0, 1, (2, 2))
        sut = BoxSpaceDiscretizer(s, np.array([[2, 2], [2, 2]]))

        self.assertEqual(sut.discretize([[0, 0], [0, 0]]), 0)
        self.assertEqual(sut.discretize([[0.05, 0], [0, 0]]), 0)
        self.assertEqual(sut.discretize([[0.95, 0], [0, 0]]), 8)
        self.assertEqual(sut.discretize([[0, 0.05], [0, 0]]), 0)
        self.assertEqual(sut.discretize([[0, 0.95], [0, 0]]), 4)
        self.assertEqual(sut.discretize([[0, 0], [0.05, 0]]), 0)
        self.assertEqual(sut.discretize([[0, 0], [0.95, 0]]), 2)
        self.assertEqual(sut.discretize([[0, 0], [0, 0.05]]), 0)
        self.assertEqual(sut.discretize([[0, 0], [0, 0.95]]), 1)
        self.assertEqual(sut.discretize([[0.1, 0.6], [0.5, 0.2]]), 6)
        self.assertEqual(sut.discretize([[1, 1], [1, 1]]), 15)
