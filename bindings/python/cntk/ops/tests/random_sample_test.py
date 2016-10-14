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
    (np.full((4), 42),                                                               1,     True,   np.full((4), 1/4),                                    0.0001),
    (np.full((4), 42),                                                              13,     True,   np.full((4), 13/4),                                   0.0001),
    ([1,2,3],                                                                       42,     True,   [42/(1+2+3), 2*42/(1+2+3), 3*42/(1+2+3)],             0.0001),
    (np.full((4), 42),                                                               1,     False,  np.full((4), 1/4),                                    0.0001),
    # Use 300 weights where the first 200 hundred weights are high compared to the rest. Sample 200 without replacement. 
    (np.concatenate((np.full((100),100),np.full((100),10),np.full((100),0.1))),    200,     False,  np.concatenate((np.full((200),1),np.full((100),0))),  0.05),
    # Having more classes than samples is not allowed when sampling without replacment. Check if exception is thrown.
    (np.full((4), 42),                                                              50,     False,  np.full((4), 13/4),                                   0.0001), 
]

@pytest.mark.parametrize("weights, num_samples, allow_duplicates, expected, tolerance", TEST_CASES)
def test_random_sample_inclusion_frequency(weights, num_samples, allow_duplicates, expected, tolerance, device_id, precision):

    result = random_sample_inclusion_frequency(weights, num_samples, allow_duplicates)

    if num_samples >= len(weights) and not allow_duplicates:
        # in case num_samples => len(weights) we expect an exception to be thrown
        with pytest.raises(RuntimeError):
            result.eval()
    else:
        assert np.allclose(result.eval(), expected, atol=tolerance)
