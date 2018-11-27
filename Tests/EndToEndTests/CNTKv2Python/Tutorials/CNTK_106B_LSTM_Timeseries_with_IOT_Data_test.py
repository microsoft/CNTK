# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(abs_path)

from nb_helper import get_output_stream_from_cell

notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_106B_LSTM_Timeseries_with_IOT_Data.ipynb")
notebook_timeoutSeconds = 1800

def test_cntk_106B_lstm_timeseries_with_iot_data_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalErrorByDeviceId = { -1: 0.000076, 0: 0.000076 }

def test_cntk_106B_lstm_timeseries_with_iot_data_evalCorrect(nb, device_id):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Print the test error', cell.source)]
    assert len(testCell) == 1
    text = get_output_stream_from_cell(testCell[0])
    m = re.match(r"mse for test: (?P<actualEvalError>\d+\.\d+)\r?$", text)
    assert np.isclose(float(m.group('actualEvalError')), expectedEvalErrorByDeviceId[device_id], atol=0.00002)
