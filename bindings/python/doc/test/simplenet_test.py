# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from simplenet import ffnet

TOLERANCE_ABSOLUTE = 5E-2

def test_ffnet_error(device_id):
    np.random.seed(98052)
    last_avg_error, avg_error = ffnet()
    expected_last_avg_error = 0.24
    expected_avg_error = 0.04

    assert np.allclose(avg_error, expected_avg_error, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(last_avg_error, expected_last_avg_error, atol=TOLERANCE_ABSOLUTE)
