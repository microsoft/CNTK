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
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_200_GuidedTour.ipynb")
  
    return notebook

# Test only on GPU, since it is too slow on CPU.
notebook_deviceIdsToRun = [0]

def test_cntk_200_guidedtour_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalError = '8.11%'

def test_cntk_200_guidedtour_evalCorrect(nb):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('test_metric_lr =', cell.source)]
    assert len(testCell) == 1
    executeResultText = testCell[0].outputs[0]['text']
    print(executeResultText)
    assert re.search(expectedEvalError, executeResultText)
