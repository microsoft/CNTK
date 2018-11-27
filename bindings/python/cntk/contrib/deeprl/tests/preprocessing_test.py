# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import unittest

import numpy as np

from cntk.contrib.deeprl.agent.shared.preprocessing import AtariPreprocessing


class AtariPreprocessingTest(unittest.TestCase):
    """Unit tests for AtariPreprocessing."""

    def test_atari_preprocessing(self):
        p = AtariPreprocessing((210, 160, 3), 4)
        self.assertEqual(p._AtariPreprocessing__history_len, 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__previous_raw_image,
            np.zeros((210, 160, 3), dtype='uint8'))
        self.assertEqual(len(p._AtariPreprocessing__processed_image_seq), 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[0],
            np.zeros((84, 84), dtype='uint8'))
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[-1],
            np.zeros((84, 84), dtype='uint8'))

        r = p.preprocess(np.ones((210, 160, 3), dtype=np.uint8))
        np.testing.assert_array_equal(
            p._AtariPreprocessing__previous_raw_image,
            np.ones((210, 160, 3), dtype=np.uint8))
        self.assertEqual(len(p._AtariPreprocessing__processed_image_seq), 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[0],
            np.zeros((84, 84), dtype='uint8'))
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[-1],
            np.ones((84, 84), dtype='uint8'))
        self.assertEqual(r.shape, (4, 84, 84))
        np.testing.assert_array_equal(
            np.squeeze(r[3, :, :]),
            np.ones((84, 84), dtype='uint8'))

        p.preprocess(np.ones((210, 160, 3), dtype=np.uint8) * 2)
        p.preprocess(np.ones((210, 160, 3), dtype=np.uint8) * 3)
        r = p.preprocess(np.ones((210, 160, 3), dtype=np.uint8) * 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__previous_raw_image,
            np.ones((210, 160, 3), dtype='uint8') * 4)
        self.assertEqual(len(p._AtariPreprocessing__processed_image_seq), 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[0],
            np.ones((84, 84), dtype='uint8'))
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[1],
            np.ones((84, 84), dtype='uint8') * 2)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[2],
            np.ones((84, 84), dtype='uint8') * 3)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[3],
            np.ones((84, 84), dtype='uint8') * 4)
        self.assertEqual(r.shape, (4, 84, 84))
        np.testing.assert_array_equal(
            np.squeeze(r[3, :, :]),
            np.ones((84, 84), dtype='uint8') * 4)

        p.reset()
        np.testing.assert_array_equal(
            p._AtariPreprocessing__previous_raw_image,
            np.zeros((210, 160, 3), dtype='uint8'))
        self.assertEqual(len(p._AtariPreprocessing__processed_image_seq), 4)
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[0],
            np.zeros((84, 84), dtype='uint8'))
        np.testing.assert_array_equal(
            p._AtariPreprocessing__processed_image_seq[-1],
            np.zeros((84, 84), dtype='uint8'))
