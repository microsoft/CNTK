# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np

from _cntk_py import force_deterministic_algorithms

force_deterministic_algorithms()
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_301_Image_Recognition_with_Deep_Transfer_Learning.ipynb")

notebook_timeoutSeconds = 900
# Runs on GPU only, batch normalization training on CPU is not yet implemented.
notebook_deviceIdsToRun = [0]


def test_CNTK_301_Image_Recognition_with_Deep_Transfer_Learning_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

# Testing only for functional correctness for now. TODO: Revisit post RC
#expectedEvalError = 70.0
#expectedEvalErrorAtol = 10.0    

#def test_CNTK_301_Image_Recognition_with_Deep_Transfer_Learning_predictionerror(nb):
#    testCell = [cell for cell in nb.cells
#                if cell.cell_type == 'code' and re.search('# Test: Accuracy on flower data', cell.source)]
#   assert len(testCell) == 1
#   print(testCell[0].outputs[0])
#   m = re.match(r'Prediction accuracy: (?P<actualEvalError>\d+\.\d+)%', testCell[0].outputs[0]['text'])
#   assert np.isclose(float(m.group('actualEvalError')), expectedEvalError, atol=expectedEvalErrorAtol)