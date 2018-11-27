﻿# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk import distributed
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "SequenceToSequence", "CMUDict", "Python")
sys.path.append(abs_path)
sys.path.append(example_dir)

from distributed_common import mpiexec_test
from prepare_test_data import cmudict_dataset_directory
script_under_test = os.path.join(example_dir, "Sequence2Sequence_Distributed.py")

mpiexec_params = [ "-n", "2"]

def test_sequence_to_sequence_distributed_gpu(device_id):
    params = [ "-e", "2",
               "-datadir", cmudict_dataset_directory(),
               "-q", "1",
               "-ms", "100",
               "-es", "500",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.8625, False, 0, 2E-2)

def test_sequence_to_sequence_distributed_block_momentum(device_id):
    params = [ "-e", "4",
               "-datadir", cmudict_dataset_directory(),
               "-ms", "100",
               "-es", "1000",
               "-b", "3200",
               "-device", str(device_id) ]
    mpiexec_test(device_id, script_under_test, mpiexec_params, params, 0.8612, False, 1, 2E-2)
