# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
from cntk.utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
from cntk.io import ReaderConfig, ImageDeserializer
import pytest

from examples.CifarResNet.CifarResNet import cifar_resnet

TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_resnet_error(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    set_default_device(cntk_device(device_id))

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
    except KeyError:
        base_path = os.path.join(
            *"../../../../Examples/Image/Datasets/CIFAR-10/cifar-10-batches-py".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))

    test_error = cifar_resnet(base_path)
    expected_test_error = 0.5

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
