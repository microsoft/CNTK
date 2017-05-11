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

mpiexec_params = [ "-n", "2"]
TIMEOUT_SECONDS = 30

def test_finalize_with_exception_no_hang():
    # Starting main function below
    cmd = ["mpiexec"] + mpiexec_params + ["python", __file__]
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
