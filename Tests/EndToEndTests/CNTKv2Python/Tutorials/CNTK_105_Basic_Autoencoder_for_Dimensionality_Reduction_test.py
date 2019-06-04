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
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.ipynb")

TOLERANCE_ABSOLUTE = 1E-1

def test_cntk_105_basic_autoencoder_for_dimensionality_reduction_noErrors(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

expectedError = 3.05

def test_cntk_105_basic_autoencoder_for_dimensionality_reduction_simple_trainerror(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Simple autoencoder test error', cell.source)]
    assert np.isclose(float((testCell[0].outputs[0])['text']), expectedError, atol = TOLERANCE_ABSOLUTE)
