# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
from . import _all_close_or_less
import pytest

@pytest.fixture(scope="module")
def notebook_path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_106A_LSTM_Timeseries_with_Simulated_Data.ipynb")
  
    return notebook

def test_cntk_106A_lstm_timeseries_with_simulated_data_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalErrorByDeviceId = { -1: 0.000049, 0: 0.000049 }

def test_cntk_106A_lstm_timeseries_with_simulated_data_evalCorrect(nb, device_id):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Print validate and test error', cell.source)]
    assert len(testCell) == 1
    m = re.match(r"mse for test: (?P<actualEvalError>\d+\.\d+)\r?$", testCell[0].outputs[0]['text'])
    assert _all_close_or_less(float(m.group('actualEvalError')), expectedEvalErrorByDeviceId[device_id], atol=0.000004)
