# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_206A_IOT_Example_with_LSTM.ipynb")

def test_cntk_206A_iot_example_with_LSTM_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalErrorByDeviceId = { -1: 0.0001, 0: 0.0001 }

def test_cntk_206A_iot_example_with_LSTM_evalCorrect(nb, device_id):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Print validate and test error', cell.source)]
    assert len(testCell) == 1
    print(testCell[0].outputs[0]['text'])
    m = re.match(r"msre for test: (?P<actualEvalError>\d+\.\d+)\r?$", testCell[0].outputs[0]['text'])
    print(m)
    print(float(m.group('actualEvalError')))
    # TODO tighten tolerances
    assert np.isclose(float(m.group('actualEvalError')), expectedEvalErrorByDeviceId[device_id], atol=0.000001)
