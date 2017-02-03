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
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python")
script_under_test = os.path.join(example_dir, "ConvNet_CIFAR10_DataAug_Distributed.py")

sys.path.append(example_dir)

TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

def data_set_directory():
    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
        # N.B. CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY has {train,test}_map.txt
        #      and CIFAR-10_mean.xml in the base_path.
    except KeyError:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))
    return base_path

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
    results = re.findall("Final Results: Minibatch\[.+?\]: errs = (.+?)%", str_out)

    assert len(results) == 2

    if match_exactly:
        assert results[0] == results[1]
    else:
        assert np.allclose(float(results[0]), float(results[1]), atol=per_minibatch_tolerance)

    assert np.allclose(float(results[0])/100, expected_test_error, atol=error_tolerance)

def test_cifar_convnet_distributed(device_id):
    params = [ "-e", "2",
               "-d", data_set_directory(),
               "-q", "32",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.617)

def test_cifar_convnet_distributed_1bitsgd(device_id):
    params = [ "-e", "2",
               "-d", data_set_directory(),
               "-q", "1",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.617)


def test_cifar_convnet_distributed_block_momentum(device_id):
    params = [ "-e", "2",
               "-d", data_set_directory(),
               "-b", "3200",
               "-device", "0" ]
    mpiexec_test(device_id, script_under_test, params, 0.6457, False, 10)
