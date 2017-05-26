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
from .ops_test_utils import AA, precision
from cntk import random_sample_inclusion_frequency, random_sample, times

INCLUSION_FREQUENCY_TEST_CASES = [
    # drawing 1 sample
    (np.full((4), 42.), 1, True, np.full((4), 1 / 4), 0.0001, False),

    # drawing 13 samples
    (np.full((4), 42.), 13,
     True,   np.full((4), 13 / 4), 0.0001, False),

    # drawing more samples than there are classes
    ([1., 2., 3.], 42, True, [42 / (1 + 2 + 3), 2 * 42 / \
                              (1 + 2 + 3), 3 * 42 / (1 + 2 + 3)], 0.0001, False),

    # Use 300 weights where the first 200 hundred weights are high compared to
    # the rest. Sample 200 without replacement.
    (np.concatenate((np.full((100), 100), np.full((100), 10), np.full((100), 0.1))), 200,
     False, np.concatenate((np.full((200), 1), np.full((100), 0))),  0.1,   False),

    # Having more classes than samples is not allowed when sampling without
    # replacment. Check if exception is thrown.
    (np.full((4), 42.), 50,
     False,  np.full((4), 13 / 4), 0.0001, True),

    # Number of requested samples must be positive
    ([1., 2., 3.], 0, False,  np.full((4), 13 / 4), 0.0001, True),

    # Negative sampling weigts are not allowed.
    ([1, -1.], 1,
     True, [0], 0.0001, True),
    ([1, -1.], 1, False, [0], 0.0001, True),
]


@pytest.mark.parametrize("weights, num_samples, allow_duplicates, expected, tolerance, raises_exception", INCLUSION_FREQUENCY_TEST_CASES)
def test_random_sample_inclusion_frequency(weights, num_samples, allow_duplicates, expected, tolerance, raises_exception, device_id, precision):

    weights = AA(weights, precision)

    if raises_exception:
        with pytest.raises(ValueError):
            result = random_sample_inclusion_frequency(
                weights, num_samples, allow_duplicates)
            result.eval()
    else:
        result = random_sample_inclusion_frequency(
            weights, num_samples, allow_duplicates)
        assert np.allclose(result.eval(), expected, atol=tolerance)

RANDOM_SAMPLE_TEST_CASES_WITH_REPLACEMENT = [
    ([1., 3., 5., 1.],  1000, 0.05, False),
    ([1., -1.],  100, 0.0, True),
]


@pytest.mark.parametrize("weights, num_samples,  tolerance, raises_exception", RANDOM_SAMPLE_TEST_CASES_WITH_REPLACEMENT)
def test_random_sample_with_replacement(weights, num_samples, tolerance, raises_exception, device_id, precision):

    weights = AA(weights, precision)
    expected_relative_frequency = weights / np.sum(weights)
    num_calls = 10
    identity = np.identity(weights.size)
    allow_duplicates = True  # sample with replacement

    if raises_exception:
        with pytest.raises(ValueError):
            result = random_sample(weights, num_samples, allow_duplicates)
            result.eval()
    else:
        observed_frequency = np.empty_like(weights)
        for i in range(0, num_calls):
            result = random_sample(weights, num_samples, allow_duplicates)
            denseResult = times(result, identity)
            observed_frequency += np.sum(denseResult.eval(), 0)
        observed_relative_frequency = observed_frequency / \
            (num_calls * num_samples)
        assert np.allclose(observed_relative_frequency,
                           expected_relative_frequency, atol=tolerance)


RANDOM_SAMPLE_TEST_CASES_WITHOUT_REPLACEMENT = [
    ([1., 3, 50., 1., 0.], 4, (1, 1, 1, 1, 0), 0.0, False),
    ([1., -1.],  1, None,   0.0, True),
]


@pytest.mark.parametrize("weights, num_samples, expected_count, tolerance, raises_exception", RANDOM_SAMPLE_TEST_CASES_WITHOUT_REPLACEMENT)
def test_random_sample_without_replacement(weights, num_samples, expected_count, tolerance, raises_exception, device_id, precision):

    weights = AA(weights, precision)
    identity = np.identity(weights.size)
    allow_duplicates = False  # sample without replacement

    if raises_exception:
        with pytest.raises(ValueError):
            result = random_sample(weights, num_samples, allow_duplicates)
            result.eval()
    else:
        result = random_sample(weights, num_samples, allow_duplicates)
        denseResult = times(result, identity)
        observed_count = np.sum(denseResult.eval(), 0)
        assert np.allclose(observed_count, expected_count, atol=tolerance)

def test_random_sample_with_explicit_seed(device_id, precision):
    weights = AA([x for x in range(0, 10)], precision)
    identity = np.identity(weights.size)
    allow_duplicates = False  # sample without replacement
    num_samples = 5;
    seed = 123
    to_dense = lambda x: times(x, identity).eval()
    result1 = to_dense(random_sample(weights, num_samples, allow_duplicates, seed))
    result2 = to_dense(random_sample(weights, num_samples, allow_duplicates, seed))
    result3 = to_dense(random_sample(weights, num_samples, allow_duplicates, seed+1))
    result4 = to_dense(random_sample(weights, num_samples, allow_duplicates))
    assert np.allclose(result1, result2)
    assert not np.allclose(result1, result3)
    assert not np.allclose(result1, result4)
