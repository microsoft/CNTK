# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import pytest
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_208_Speech_Connectionist_Temporal_Classification.ipynb")
notebook_timeoutSeconds = 600
# TODO currently limited to GPU; need to investigate hangs in our Linux CI env
notebook_deviceIdsToRun = [0]

def test_cntk_208_speech_connectionist_temporal_classification_noErrors(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalError1 = '0.98'
expectedEvalError2 = '0.99'
expectedEvalError3 = '1.0'

def test_cntk_208_speech_connectionist_temporal_classification_evalCorrect(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search(r'trainer\.test_minibatch', cell.source)]
    assert len(testCell) == 1
    assert testCell[0].outputs[0]['data']['text/plain'] == expectedEvalError1 or testCell[0].outputs[0]['data']['text/plain'] == expectedEvalError2 or testCell[0].outputs[0]['data']['text/plain'] == expectedEvalError3
