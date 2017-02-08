# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.device import set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "NumpyInterop"))
from FeedForwardNet import ffnet

TOLERANCE_ABSOLUTE = 1E-03

def test_ffnet_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    set_default_device(cntk_device(device_id))

    np.random.seed(0)
    avg_error = ffnet()
    expected_avg_error = 0.04
    assert np.allclose(avg_error, expected_avg_error, atol=TOLERANCE_ABSOLUTE)
