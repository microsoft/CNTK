# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
from cntk import DeviceDescriptor
from cntk.io import ReaderConfig, ImageDeserializer

from examples.CifarResNet.CifarResNet import cifar_resnet

TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_resnet_error(device_id):
    target_device = DeviceDescriptor.gpu_device(0)
    DeviceDescriptor.set_default_device(target_device)

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
    except KeyError:
        base_path = os.path.join(
            *"../../../../Examples/Image/Miscellaneous/CIFAR-10/cifar-10-batches-py".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))

    test_error = cifar_resnet(base_path)
    expected_test_error = 0.7

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
