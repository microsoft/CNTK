# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests random sampling related operations
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type
from ...utils import sanitize_dtype_cntk, ones_like, eval
from  cntk import random_sample_inclusion_frequency, times

TEST_CASES = [
    (np.full((4), 42), 1, True, np.full((4), 1/4), 0.0001),
    (np.full((4), 42), 13, True, np.full((4), 13/4), 0.0001),
    ([1,2,3], 42, True, [42/(1+2+3), 2*42/(1+2+3), 3*42/(1+2+3)], 0.0001),
    (np.full((4), 42), 1, False, np.full((4), 1/4), 0.0001),
#   (np.full((4), 42), 13, False, np.full((4), 13/4), 0.0001), make sure we get an exception in this case!
]

@pytest.mark.parametrize("weights, num_samples, allow_duplicates, expected, tolerance", TEST_CASES)
def test_random_sample_inclusion_frequency(weights, num_samples, allow_duplicates, expected, tolerance, device_id, precision):

    result = random_sample_inclusion_frequency(weights, num_samples, allow_duplicates).eval()
    
    assert np.allclose(result, expected, atol=tolerance)

@pytest.mark.parametrize("weights, num_samples, allow_duplicates, expected, tolerance", TEST_CASES)
def test_random_sample(weights, num_samples, allow_duplicates, expected, tolerance, device_id, precision):

    samples = random_sample_inclusion_frequency(weights, num_samples, allow_duplicates)

    
    assert np.allclose(result, expected, atol=tolerance)
