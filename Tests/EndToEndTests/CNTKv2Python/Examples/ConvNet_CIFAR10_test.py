# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.device import try_set_default_device, gpu

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python" ))

from ConvNet_CIFAR10 import convnet_cifar10

TOLERANCE_ABSOLUTE = 1E-1

def test_convnet_cifar_error(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    error = convnet_cifar10(epoch_size=2000, minibatch_size=32, max_epochs=10)

    expected_error = 0.64
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
