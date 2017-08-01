# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import signal
import subprocess
import re
import pytest

from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "GoogLeNet", "InceptionV3", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_ImageNet_data
script_under_test = os.path.join(example_dir, "InceptionV3_ImageNet_Distributed.py")

mpiexec_params = ["-n", "2"]

def test_inception_v3_imagenet_distributed(device_id):
    # Inception-V3 distributed test on ImageNet need plenty of memory,
    # for now, the test server might feel hard to handle
    pytest.skip('Mute Inception-V3 distributed test temporarily')

    params = ["-n", "2",
              "-datadir", prepare_ImageNet_data(),
              "-q", "32",
              "-e", "200",
              "-m", "2",
              "-r",
              "-device", str(device_id)]

    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.99, True, timeout_seconds=400)
