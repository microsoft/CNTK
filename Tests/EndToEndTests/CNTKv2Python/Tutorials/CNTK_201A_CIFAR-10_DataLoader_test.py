# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
import pytest
import shutil
import pytest

@pytest.fixture(scope="module")
def notebook_path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_201A_CIFAR-10_DataLoader.ipynb")
  
    return notebook

abs_path = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "data", "CIFAR-10")
reWeekly = re.compile(r'^weekly\b', re.IGNORECASE)

# Run this on CPU only
notebook_deviceIdsToRun = [-1]

notebook_timeoutSeconds = 900

@pytest.fixture(scope='module')
def clean_data(device_id):
  # Delete the data folder if we're supposed to run
  if device_id in notebook_deviceIdsToRun and os.path.exists(datadir):
    shutil.rmtree(datadir)

@pytest.mark.skipif(not reWeekly.search(os.environ.get('TEST_TAG')),
                    reason="only runs as part of the weekly tests")
def test_cntk_201a_cifar_10_dataloader_noErrors(clean_data, nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
    assert os.path.exists(datadir)
