# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import subprocess
import signal
import re
import pytest
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Text', 'LightRNN', 'LightRNN')
sys.path.append(abs_path)
sys.path.append(example_dir)
script_under_test = os.path.join(example_dir, 'train.py')

TIMEOUT_SECONDS = 300
TOLERANCE_ABSOLUTE = 1E-1

def run_command(**kwargs):
    command = ['mpiexec', '-n', '1', 'python', script_under_test]
    for key, value in kwargs.items():
        command += ['-' + key, str(value)]
    return command

def test_lightrnn(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    expected_valid_error = 7.251514
    expected_test_error = 7.305801

    command = run_command(datadir=os.path.join(example_dir, '..', 'test'),
                          outputdir=os.path.join(example_dir, '..', 'LightRNN'),
                          vocabdir=os.path.join(example_dir, '..', 'test'),
                          vocab_file=os.path.join(example_dir, '..', 'test', 'vocab.txt'),
                          alloc_file=os.path.join(example_dir, '..', 'test', 'word-0.location'),
                          vocabsize=1566,
                          optim='adam', lr=0.20,
                          embed=500, nhid=500, batchsize=20, layer=2,
                          epochs=1)
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=TIMEOUT_SECONDS)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    results = re.findall("Epoch  1 Done : Valid error = (.+), Test error = (.+)", str_out)
    results = results[0]
    assert len(results) == 2
    assert np.allclose([float(results[0]), float(results[1])], [expected_valid_error, expected_test_error], atol=TOLERANCE_ABSOLUTE)
