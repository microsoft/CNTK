# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest

import numpy as np

from cntk.contrib.deeprl.agent.shared.cntk_utils import (huber_loss,
                                                         negative_of_entropy_with_softmax)
from cntk.ops import input_variable


class CNTKUtilsTest(unittest.TestCase):
    """Unit tests for cntk_utils."""

    def test_huber_loss(self):
        i1 = input_variable((2))
        i2 = input_variable((2))

        np.testing.assert_array_equal(
            huber_loss(i1, i2).eval({
                i1: [[2, 1], [1, 5]],
                i2: [[4, 1], [1, 4]]
            }),
            [1.5, 0.5]
        )

    def test_entropy(self):
        i = input_variable((2))

        np.testing.assert_almost_equal(
            negative_of_entropy_with_softmax(i).eval({
                i: [[0.5, 0.5], [1000, 1]]
            }),
            [-0.693147181, 0]
        )
