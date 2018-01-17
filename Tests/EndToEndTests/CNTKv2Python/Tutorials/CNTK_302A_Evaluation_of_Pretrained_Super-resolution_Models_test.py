# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
import pytest

@pytest.fixture(scope="module")
def notebook_path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_302A_Evaluation_of_Pretrained_Super-resolution_Models.ipynb")
   
    return notebook

abs_path = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "data", "BerkeleySegmentationDataset")

# Run this on GPU only
notebook_deviceIdsToRun = [0]
notebook_timeoutSeconds = 600

def test_cntk_302a_evaluation_superresolution_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
    assert os.path.exists(os.path.join(datadir, "example_images"))
    assert os.path.exists(os.path.join(datadir, "example_results"))

    assert len(os.listdir(os.path.join(datadir, "example_images"))) > 0
    assert len(os.listdir(os.path.join(datadir, "example_results"))) > 0

    results = os.path.join(datadir, "example_results")

    for folder in os.listdir(results):
        assert os.path.isdir(os.path.join(results, folder))
        assert len(os.listdir(os.path.join(results, folder))) > 0