# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import signal
import shutil
import subprocess
import re
import pytest
from cntk.train.distributed import Communicator, mpi_communicator

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from distributed_common import mpiexec_execute

def test_finalize_with_exception_no_hang():
    str_out = mpiexec_execute(__file__, ["-n", "2"], [])

    results = re.findall("Completed with exception.", str_out)
    assert len(results) == 1

    results = re.findall("Completed successfully.", str_out)
    assert len(results) == 0

if __name__=='__main__':
    communicator = mpi_communicator()
    if communicator.is_main():
        print("Completed with exception.")
        raise ValueError()
    else:
        communicator.barrier()
        print("Completed successfully.")
    Communicator.finalize()
