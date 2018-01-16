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
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "GoogLeNet", "BN-Inception", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_CIFAR10_data
script_under_test = os.path.join(example_dir, "BN_Inception_CIFAR10.py")

from BN_Inception_CIFAR10 import bn_inception_train_and_eval

# TODO this needs to be tightened, expected_error may need to be fixed, too.
#      We've seen errors of 0.754, 0.776, 0.778 on different cards / OS.
TOLERANCE_ABSOLUTE = 0.15

def test_bn_inception_cifar(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    current_path = os.getcwd()
    base_path = prepare_CIFAR10_data()
    # change dir to locate data.zip correctly
    os.chdir(base_path)

    from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)
    force_deterministic_algorithms()

    mean_data = os.path.join(base_path, 'CIFAR-10_mean.xml')
    train_data = os.path.join(base_path, 'train_map.txt')
    test_data = os.path.join(base_path, 'test_map.txt')

    try:
        error = bn_inception_train_and_eval(train_data, test_data, mean_data, minibatch_size=16, epoch_size=500,
                                    max_epochs=8, restore=False, testing_parameters=(500,16))
    finally:
        os.chdir(current_path)

    expected_error = 0.88
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
