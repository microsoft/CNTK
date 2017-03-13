# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import pytest
import subprocess

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python")
sys.path.append(example_dir)
sys.path.append(abs_path)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_CIFAR10_data

script_under_test = os.path.join(example_dir, "TrainResNet_CIFAR10_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_cifar_resnet_distributed(device_id):
    params = [ "-e", "2",
               "-datadir", prepare_CIFAR10_data(),
               "-q", "32",
               "-es", "512",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.86, False, 3)

def test_cifar_resnet_distributed_1bitsgd(device_id):
    params = [ "-e", "2",
               "-datadir", prepare_CIFAR10_data(),
               "-q", "1",
               "-es", "512",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.86, False, 3)

def test_cifar_resnet_distributed_block_momentum(device_id):
    params = [ "-e", "2",
               "-datadir", prepare_CIFAR10_data(),
               "-b", "3200",
               "-es", "512",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.89, False, 5)
