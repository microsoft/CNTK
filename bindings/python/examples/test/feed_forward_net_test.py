# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor

from examples.NumpyInterop.FeedForwardNet import ffnet

TOLERANCE_ABSOLUTE = 1E-03

def test_ffnet_error(device_id):
    from cntk.utils import cntk_device
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    avg_error = ffnet(debug_output=False)
    expected_avg_error = 0.12
    assert np.allclose(avg_error, expected_avg_error, atol=TOLERANCE_ABSOLUTE)
