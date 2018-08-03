# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import re
import pytest

@pytest.fixture(scope="module", params=['English', 'Japanese'])
def notebook_path(request):
    language = request.param
    notebook_name = "CNTK_101_LogisticRegression.ipynb"

    import os
    abs_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials")
    
    if language == 'English':
        return os.path.join(folder_path, notebook_name)
    else:
        return os.path.join(folder_path, 'Translations', language, notebook_name)

def test_cntk_101_logisticregression_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalError = '0.12'

def test_cntk_101_logisticregression_evalCorrect(nb):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('trainer\.test_minibatch', cell.source)]
    assert len(testCell) == 1
    assert testCell[0].outputs[0]['data']['text/plain'] == expectedEvalError
