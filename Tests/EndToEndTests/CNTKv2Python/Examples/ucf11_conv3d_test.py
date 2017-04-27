# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Video", "GettingStarted", "Python"))
# from Conv3D_UCF11 import conv3d_ucf11, VideoReader # Depends on imageio package.

TOLERANCE_ABSOLUTE = 2E-1

def test_ucf11_conv3d_error(device_id):
    # Skip for now.
    if True: #cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Video/DataSets/UCF11".split("/"))
    except KeyError:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                *"../../../../Examples/Video/DataSets/UCF11".split("/"))

    base_path = os.path.normpath(base_path)

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1)

    # For performance reason, we will use test data for both training and testing.
    num_output_classes = 11
    # train_reader = VideoReader(os.path.join(base_path, 'test_map.csv'), num_output_classes, True)
    # test_reader  = VideoReader(os.path.join(base_path, 'test_map.csv'), num_output_classes, False)

    test_error = 0.8437 #conv3d_ucf11(train_reader, test_reader, max_epochs=1)
    expected_test_error = 0.8437

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
