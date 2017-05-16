# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.ops.tests.ops_test_utils import cntk_device
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python" ))

from ConvNet_CIFAR10 import convnet_cifar10

TOLERANCE_ABSOLUTE = 2E-1

def test_convnet_cifar_error(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        dataset_path = os.path.join(extPath, "Image", "CIFAR", "v0")
    else:
        dataset_path = os.path.join(abs_path,  "..", "..", "..", "..", "Examples", "Image", "DataSets", "CIFAR-10")

    error = convnet_cifar10(data_path=dataset_path, epoch_size=2000, minibatch_size=32, max_epochs=10)

    expected_error = 0.7
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
