# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import pytest

@pytest.fixture(scope="module")
def notebook_path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_207_Training_with_Sampled_Softmax.ipynb")
   
    return notebook

def test_cntk_207_sampledsoftmax_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
