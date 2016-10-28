# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
from cntk.io import ReaderConfig, ImageDeserializer
import pytest

from examples.Distributed.CifarResNet_Distributed import train_and_evaluate, create_reader

TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_resnet_distributed_error(device_id, is_1bit_sgd):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    set_default_device(cntk_device(device_id))

    if not is_1bit_sgd:
        pytest.skip('test only runs in 1-bit SGD')

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
    except KeyError:
        base_path = os.path.join(
            *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))

    base_path = os.path.normpath(base_path)

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1) 
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    #force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    test_error = train_and_evaluate(base_path, total_epochs=5)
    expected_test_error = 0.5

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
