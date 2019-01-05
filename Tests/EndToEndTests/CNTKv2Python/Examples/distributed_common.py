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

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python")

sys.path.append(example_dir)

TOLERANCE_ABSOLUTE = 2E-1
TIMEOUT_SECONDS = 300

def mpiexec_execute(script, mpiexec_params, params, timeout_seconds=TIMEOUT_SECONDS, device_id=None, use_only_cpu=False):
    if device_id is not None:
        device_is_cpu = (cntk_device(device_id).type() != DeviceKind_GPU)
        if use_only_cpu != device_is_cpu:
            pytest.skip('test only runs on ' + ('CPU' if use_only_cpu else 'GPU'))

    cmd = ["mpiexec"] + mpiexec_params + ["python", script] + params
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=timeout_seconds)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    return str_out

def mpiexec_test(device_id, script, mpiexec_params, params, expected_test_error, match_exactly=True, per_minibatch_tolerance=TOLERANCE_ABSOLUTE, error_tolerance=TOLERANCE_ABSOLUTE, timeout_seconds=TIMEOUT_SECONDS, use_only_cpu=False):
    str_out = mpiexec_execute(script, mpiexec_params, params, timeout_seconds, device_id, use_only_cpu)
    results = re.findall(r"Finished Evaluation \[.+?\]: Minibatch\[.+?\]: metric = (.+?)%", str_out)

    assert len(results) == 2, str_out

    if match_exactly:
        assert results[0] == results[1], str_out
    else:
        if abs((float(results[0]) - float(results[1]))) > per_minibatch_tolerance:
            print(str_out)
            assert False
    assert np.allclose(float(results[0])/100, expected_test_error, atol=error_tolerance), str_out
