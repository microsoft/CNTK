# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import signal
import shutil
import subprocess
import re
import pytest
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python")
sys.path.append(example_dir)
script_under_test = os.path.join(example_dir, "ConvNet_CIFAR10_DataAug_Distributed.py")

from distributed_common import mpiexec_test
from prepare_test_data import prepare_CIFAR10_data

mpiexec_params = [ "-n", "2"]

def test_cifar_convnet_distributed(device_id):
    # Create a path to TensorBoard log directory and make sure it does not exist.
    abs_path = os.path.dirname(os.path.abspath(__file__))
    tb_logdir = os.path.join(abs_path, 'ConvNet_CIFAR10_DataAug_Distributed_test_log')
    if os.path.exists(tb_logdir):
        shutil.rmtree(tb_logdir)

    params = [ "-n", "2",
               "-m", "64",
               "-e", "3200",
               "-datadir", prepare_CIFAR10_data(),
               "-tensorboard_logdir", tb_logdir,
               "-q", "32",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.75, False, per_minibatch_tolerance=1e-2) # False since different workers may have different #cores

    # Ensure that the TensorBoard log directory was created and contains exactly one file with the expected name.
    tb_files = 0
    for tb_file in os.listdir(tb_logdir):
        assert tb_file.startswith("events.out.tfevents")
        tb_files += 1
    assert tb_files == 1

def test_cifar_convnet_distributed_1bitsgd(device_id):
    params = [ "-n", "2",
               "-m", "64",
               "-e", "3200",
               "-datadir", prepare_CIFAR10_data(),
               "-q", "1",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.75, False, per_minibatch_tolerance=1e-2)

def test_cifar_convnet_distributed_block_momentum(device_id):
    params = [ "-n", "2",
               "-m", "64",
               "-e", "3200",
               "-datadir", prepare_CIFAR10_data(),
               "-b", "1600",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.78, False, 10)
