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


abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Speech", "AN4", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test, mpiexec_execute
from prepare_test_data import an4_dataset_directory

script_under_test = os.path.join(example_dir, "HTK_LSTM_Truncated_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_htk_lstm_truncated_distributed_gpu(device_id):
    params = [ "-n", "3",
               "-datadir", an4_dataset_directory(),
               "-q", "1",
               "-m", "640",
               "-e", "1000",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.76, True)

def test_htk_lstm_truncated_distributed_block_momentum(device_id):

    params = [ "-n", "3",
               "-m", "640",
               "-e", "1000",
               "-datadir", an4_dataset_directory(),
               "-b", "1600",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.76, False, 4)


def test_htk_lstm_truncated_distributed_gpu_with_cv(device_id):
    # Make sure that full sequence cross validation
    # works in the middle of bptt training
    params = [ "-n", "2",
               "-datadir", an4_dataset_directory(),
               "-q", "1",
               "-m", "640",
               "-e", "1500",
               "-cvfreq", "1000",
               "-device", str(device_id) ]

    output = mpiexec_execute(device_id=device_id, script=script_under_test, mpiexec_params=mpiexec_params, params=params)
    results = re.findall(r"Finished Evaluation \[.+?\]: Minibatch\[.+?\]: metric = (.+?)%", output)
    assert len(results) == 6, output

