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
from cntk.device import set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python")
sys.path.append(example_dir)
from prepare_test_data import prepare_CIFAR10_data
script_under_test = os.path.join(example_dir, "ConvNet_CIFAR10_DataAug_Distributed.py")

TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

def mpiexec_test(device_id, script, params, expected_test_error, match_exactly=True, per_minibatch_tolerance=TOLERANCE_ABSOLUTE, error_tolerance=TOLERANCE_ABSOLUTE):
    if cntk_device(device_id).type() != DeviceKind_GPU:
       pytest.skip('test only runs on GPU')

    cmd = ["mpiexec", "-n", "2", "python", script] + params
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    results = re.findall("Finished Evaluation \[.+?\]: Minibatch\[.+?\]: metric = (.+?)%", str_out)

    assert len(results) == 2
    print(results)

    if match_exactly:
        assert results[0] == results[1]
    else:
        assert np.allclose(float(results[0]), float(results[1]), atol=per_minibatch_tolerance)

    assert np.allclose(float(results[0])/100, expected_test_error, atol=error_tolerance)

def test_cifar_convnet_distributed(device_id):
    params = [ "-n", "2",
               "-m", "64", 
               "-e", "3200",
               "-datadir", prepare_CIFAR10_data(),
               "-q", "32",
               "-r",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.75, True)

def test_cifar_convnet_distributed_1bitsgd(device_id):
    params = [ "-n", "2",
               "-m", "64", 
               "-e", "3200", 
               "-datadir", prepare_CIFAR10_data(),
               "-q", "1",
               "-r",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.75, True)


def test_cifar_convnet_distributed_block_momentum(device_id):
    params = [ "-n", "2",
               "-m", "64", 
               "-e", "3200",
               "-datadir", prepare_CIFAR10_data(),
               "-b", "1600",
               "-r",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.78, False, 10)
