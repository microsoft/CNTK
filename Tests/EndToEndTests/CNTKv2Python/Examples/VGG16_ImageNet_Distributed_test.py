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
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "VGG", "Python")
sys.path.append(example_dir)
from prepare_test_data import prepare_ImageNet_data
from distributed_common import mpiexec_test
script_under_test = os.path.join(example_dir, "VGG16_ImageNet_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_VGG16_imagenet_distributed(device_id):
    params = [ "-n", "2",
               "-m", "2",
               "-e", "2",
               "-datadir", prepare_ImageNet_data(),
               "-q", "32",
               "-device", str(device_id),
               "-r",
               "-testing"]

    # Currently we only test for CPU since the memory usage is very high for GPU (~6 GB)
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.99, True, timeout_seconds=500, use_only_cpu=True)
