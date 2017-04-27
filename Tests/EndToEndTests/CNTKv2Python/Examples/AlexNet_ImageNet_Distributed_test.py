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
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "AlexNet", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_ImageNet_data
script_under_test = os.path.join(example_dir, "AlexNet_ImageNet_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_alexnet_imagenet_distributed(device_id):
    params = [ "-n", "2",
               "-datadir", prepare_ImageNet_data(),
               "-q", "32",
               "-m", "8",
               "-e", "16",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.99, True)

def test_alexnet_imagenet_distributed_1bitsgd(device_id):
    params = [ "-n", "2",
               "-datadir", prepare_ImageNet_data(),
               "-q", "1",
               "-m", "8",
               "-e", "16",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.99, True)

# TODO Flaky on our linux machines?
def disabled_test_alexnet_imagenet_distributed_block_momentum(device_id):
    params = [ "-n", "2",
               "-m", "8",
               "-e", "16",
               "-datadir", prepare_ImageNet_data(),
               "-b", "1600",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.99, False)
