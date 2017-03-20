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

def mpiexec_test(device_id, script, mpiexec_params, params, expected_test_error, match_exactly=True, per_minibatch_tolerance=TOLERANCE_ABSOLUTE, error_tolerance=TOLERANCE_ABSOLUTE):
    if cntk_device(device_id).type() != DeviceKind_GPU:
       pytest.skip('test only runs on GPU')

    cmd = ["mpiexec"] + mpiexec_params + ["python", script] + params
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

    if match_exactly:
        assert results[0] == results[1]
    else:
        assert np.allclose(float(results[0]), float(results[1]), atol=per_minibatch_tolerance)

    assert np.allclose(float(results[0])/100, expected_test_error, atol=error_tolerance)
