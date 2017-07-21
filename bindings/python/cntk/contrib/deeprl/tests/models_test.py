# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest

from cntk.contrib.deeprl.agent.shared.models import Models


class ModelsTest(unittest.TestCase):
    """Unit tests for Models."""

    def test_parse_dueling_network_structure(self):
        a, b, c =\
            Models._parse_dueling_network_structure(
                    "[1, 2, [3], [4, 5]]")
        self.assertEqual(a, [1, 2])
        self.assertIsInstance(a[0], int)
        self.assertEqual(b, [3])
        self.assertEqual(c, [4, 5])

        a, b, c =\
            Models._parse_dueling_network_structure(
                "[None, [3], [None]]")
        self.assertEqual(a, [])
        self.assertEqual(b, [3])
        self.assertEqual(c, [])
