# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import pytest
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_200_GuidedTour.ipynb")
# Test only on GPU, since it is too slow on CPU.
notebook_deviceIdsToRun = [0]

def test_cntk_200_guidedtour_noErrors(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalError = '8.11%'

def test_cntk_200_guidedtour_evalCorrect(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('test_metric_lr =', cell.source)]
    assert len(testCell) == 1
    executeResultText = testCell[0].outputs[0]['text']
    print(executeResultText)
    assert re.search(expectedEvalError, executeResultText)
