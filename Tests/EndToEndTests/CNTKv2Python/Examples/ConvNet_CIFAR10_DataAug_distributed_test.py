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
from run_ConvNet_CIFAR10_DataAug_Distributed import run_cifar_convnet_distributed

#TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

def test_cifar_convnet_distributed_mpiexec(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')

    cmd = ["mpiexec", "-n", "2", "python", os.path.join(abs_path, "run_ConvNet_CIFAR10_DataAug_Distributed.py")]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        # TODO add timeout for Py2?
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
    assert results[0] == results[1]

# We are removing tolerance in error because running small epoch size has huge variance in accuracy. Will add
# tolerance back once convolution operator is determinsitic. 
    
#    expected_test_error = 0.617
#    assert np.allclose(float(results[0])/100, expected_test_error,
#                       atol=TOLERANCE_ABSOLUTE)