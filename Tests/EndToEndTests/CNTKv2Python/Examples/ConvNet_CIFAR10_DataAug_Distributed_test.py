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

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python")
sys.path.append(example_dir)
script_under_test = os.path.join(example_dir, "ConvNet_CIFAR10_DataAug_Distributed.py")

from distributed_common import mpiexec_test, mpiexec_execute
from prepare_test_data import prepare_CIFAR10_data

base_path = prepare_CIFAR10_data()
# change dir to locate data.zip correctly
os.chdir(base_path)

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
               "-datadir", base_path,
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

def test_cifar_convnet_distributed_gpu(device_id):
    params = [ "-n", "2",
               "-m", "64",
               "-e", "3200",
               "-datadir", base_path,
               "-q", "1",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.75, False, per_minibatch_tolerance=1e-2)

def test_cifar_convnet_distributed_block_momentum(device_id):
    params = [ "-n", "2",
               "-m", "64",
               "-e", "3200",
               "-datadir", base_path,
               "-b", "1600",
               "-r",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.78, False, 10)

def test_cifar_convnet_distributed_block_momentum(device_id):
    params = [ "-n", "1",
               "-m", "64",
               "-e", "13000",
               "-datadir", base_path,
               "-b", "1600",
               "-r",
               "-device", str(device_id) ]
    # 13000 samples / 2 worker / 64 mb_size = 101 minibatchs. 
    # We expect to see only Minibatch[ 1 -100] 
    output = mpiexec_execute(script_under_test, mpiexec_params, params, device_id=device_id)
    results = re.findall(r"Minibatch\[(.+?)\]: loss = .+?%", output)
    assert len(results) == 2
    assert results[0] == '   1- 100'
    assert results[1] == '   1- 100'
