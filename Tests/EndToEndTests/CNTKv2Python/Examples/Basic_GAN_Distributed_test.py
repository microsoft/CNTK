# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import pytest
import subprocess
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "GAN")
sys.path.append(example_dir)
sys.path.append(abs_path)

from distributed_common import mpiexec_execute
from prepare_test_data import prepare_MNIST_data

base_path = prepare_MNIST_data()
os.chdir(base_path)
script_under_test = os.path.join(example_dir, "Basic_GAN_Distributed.py")
mpiexec_params = [ "-n", "4"]

def test_cifar_resnet_distributed(device_id):
    params = [ "-datadir", base_path]
    str_out = mpiexec_execute(script_under_test, mpiexec_params, params)

    #Training loss of the generator at worker: {0} is: {2.201804}, time taken is: {40} seconds
    results = re.findall(r"Training loss of the generator at worker: \{.+?\} is: \{.+?\}", str_out)
    assert(len(results) == 4)
