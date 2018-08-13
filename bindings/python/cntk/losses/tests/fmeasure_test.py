# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the fmeasure class.
"""

import numpy as np
import cntk as C
from _cntk_py import set_fixed_random_seed

import pytest
from os import environ
@pytest.mark.skipif(environ.get('TEST_TAG') is not None and environ['TEST_TAG'] in {'weekly', 'nightly'}, reason="Temporarily disabled this test in the Nightly/Weekly builds due to random failures.")
def test_fmeasure():
    a = np.array([[[[1., 1., 1., 0., 0.],
                    [1., 1., 1., 0., 0.],
                    [1., 1., 1., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]]]], dtype=np.float32)

    b = np.array([[[[1., 1., 1., 0., 0.],
                    [1., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 0., 1.]]]], dtype=np.float32)

    set_fixed_random_seed(1)
    input_dim = (1, 5, 5)

    input_tensor = C.input_variable(input_dim)
    target_tensor = C.input_variable(input_dim)
    z = C.fmeasure(input_tensor, target_tensor)

    score = z.eval({input_tensor: a, target_tensor: b})
    FMEASURE_EXPECTED_VALUES = [[[[0.5]]]]

    assert np.allclose(score, FMEASURE_EXPECTED_VALUES)
