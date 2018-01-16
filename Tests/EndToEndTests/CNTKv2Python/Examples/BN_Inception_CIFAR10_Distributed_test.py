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
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "GoogLeNet", "BN-Inception", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_CIFAR10_data
script_under_test = os.path.join(example_dir, "BN_Inception_CIFAR10_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_bn_inception_cifar_distributed(device_id):
    params = [ "-n", "8",
               "-datadir", prepare_CIFAR10_data(),
               "-q", "32",
               "-e", "500",
               "-m", "16",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.88, True)
