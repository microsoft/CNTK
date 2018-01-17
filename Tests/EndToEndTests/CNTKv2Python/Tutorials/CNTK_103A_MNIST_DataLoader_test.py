# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "DataSets", "MNIST")
reWeekly = re.compile(r'^weekly\b', re.IGNORECASE)

# Run this on CPU only
notebook_deviceIdsToRun = [-1]

@pytest.fixture(scope="module", params=['English', 'Japanese'])
def notebook_path(request):
    language = request.param
    notebook_name = "CNTK_103A_MNIST_DataLoader.ipynb"

    import os
    abs_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials")
    
    if language == 'English':
        return os.path.join(folder_path, notebook_name)
    else:
        return os.path.join(folder_path, 'Translations', language, notebook_name)

@pytest.fixture(scope='module')
def clean_data(device_id):
  # Delete the data folder if we're supposed to run
  if device_id in notebook_deviceIdsToRun:
    import subprocess
    args = ["git", "clean", "-fdx", datadir]
    subprocess.check_call(args)

@pytest.mark.skipif(not reWeekly.search(os.environ.get('TEST_TAG')),
                    reason="only runs as part of the weekly tests")
def test_cntk_103a_mnist_dataloader_noErrors(clean_data, nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
    assert os.path.exists(os.path.join(datadir, "Test-28x28_cntk_text.txt"))
    assert os.path.exists(os.path.join(datadir, "Train-28x28_cntk_text.txt"))
