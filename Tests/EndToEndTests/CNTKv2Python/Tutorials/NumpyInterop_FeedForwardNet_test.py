# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import shutil
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "NumpyInterop"))

from FeedForwardNet import ffnet

TOLERANCE_ABSOLUTE = 1E-1

def test_numpyinterop_feedforwardnet_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    test_error = ffnet()
    expected_test_error = 0.04

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)