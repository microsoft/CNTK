# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.device import set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "MLP", "Python"))
from SimpleMNIST import simple_mnist

TOLERANCE_ABSOLUTE = 1E-1

def test_simple_mnist_error(device_id):
    from cntk.utils import cntk_device
    set_default_device(cntk_device(device_id))

    test_error = simple_mnist()
    expected_test_error = 0.09

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
