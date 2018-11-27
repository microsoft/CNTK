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
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "GoogLeNet", "InceptionV3", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import prepare_ImageNet_data

from InceptionV3_ImageNet import inception_v3_train_and_eval

TOLERANCE_ABSOLUTE = 1e-1

def test_inception_v3_imagenet(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    current_path = os.getcwd()
    base_path = prepare_ImageNet_data()
    # change dir to locate data.zip correctly
    os.chdir(base_path)

    from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)
    force_deterministic_algorithms()

    train_data = os.path.join(base_path, 'train_map.txt')
    test_data = os.path.join(base_path, 'val_map.txt')

    try:
        error = inception_v3_train_and_eval(train_data, test_data, minibatch_size=8, epoch_size=200,
                                            max_epochs=4, restore=False, testing_parameters=(200, 8))
    finally:
        os.chdir(current_path)

    expected_error = 0.99
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
