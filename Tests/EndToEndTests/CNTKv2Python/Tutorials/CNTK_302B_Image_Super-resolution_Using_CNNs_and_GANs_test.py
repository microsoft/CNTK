# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import pytest
import re
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_302B_Image_Super-resolution_Using_CNNs_and_GANs.ipynb")
datadir = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "data", "BerkeleySegmentationDataset")

# Run this on GPU only
notebook_deviceIdsToRun = [0]
notebook_timeoutSeconds = 1800

def clean_data(device_id):
  # Delete the data folder if we're supposed to run
  if device_id in notebook_deviceIdsToRun:
    import subprocess
    args = ["git", "clean", "-fdx", datadir]
    subprocess.check_call(args)

def test_cntk_302b_superresolution_cnns_gans_noErrors(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
    assert os.path.exists(os.path.join(datadir, "tests"))
    assert os.path.exists(os.path.join(datadir, "Images"))
    assert os.path.exists(os.path.join(datadir, "train64_LR"))
    assert os.path.exists(os.path.join(datadir, "train64_HR"))
    assert os.path.exists(os.path.join(datadir, "train112"))
    assert os.path.exists(os.path.join(datadir, "train224"))

    assert len(os.listdir(os.path.join(datadir, "tests"))) > 0
    assert len(os.listdir(os.path.join(datadir, "train64_LR"))) > 20000
    assert len(os.listdir(os.path.join(datadir, "train64_HR"))) == len(os.listdir(os.path.join(datadir, "train64_LR")))
    assert len(os.listdir(os.path.join(datadir, "train112"))) > 10000
    assert len(os.listdir(os.path.join(datadir, "train224"))) == len(os.listdir(os.path.join(datadir, "train112")))