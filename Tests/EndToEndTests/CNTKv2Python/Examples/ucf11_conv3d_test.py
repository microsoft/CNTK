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
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Video", "GettingStarted", "Python"))
from Conv3D_UCF11 import conv3d_ucf11, VideoReader # Depends on imageio package.

from prepare_test_data import prepare_UCF11_data

TOLERANCE_ABSOLUTE = 2E-1

#@pytest.mark.skipif(sys.platform != 'win32',
#                    reason="does currently run only on Windows")
@pytest.mark.skip(reason="Issue with imageio.plugins.ffmpeg.Reader. Error: Couldn't load meta information.")
def test_ucf11_conv3d_error(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    prepare_UCF11_data()

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Video/DataSets/UCF11".split("/"))

    base_path = os.path.normpath(base_path)

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1)

    num_output_classes = 11
    train_reader = VideoReader(os.path.join(base_path, 'train_map.csv'), num_output_classes, True, 100)
    test_reader  = VideoReader(os.path.join(base_path, 'test_map.csv'), num_output_classes, False, 40)

    test_error = conv3d_ucf11(train_reader, test_reader, max_epochs=1)
    expected_test_error = 0.8

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
