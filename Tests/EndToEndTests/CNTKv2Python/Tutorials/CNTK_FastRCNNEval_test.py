# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import sys
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Examples","Image","Detection","FastRCNN", "BrainScript", "CNTK_FastRCNN_Eval.ipynb")

sys.path.append(abs_path)

from nb_helper import get_output_stream_from_cell

# For now the test only supported on linux with python 3.4
@pytest.mark.skipif(not (sys.version_info.major == 3 and sys.version_info.minor == 4),
                    reason="requires python 3.4")
@pytest.mark.skipif(sys.platform == 'win32',
                    reason="does not currently run on windows")
def test_cntk_fastrcnn_eval_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]

    assert errors == []

# For now the test only supported on linux with python 3.4
@pytest.mark.skipif(not (sys.version_info.major == 3 and sys.version_info.minor == 4),
                    reason="requires python 3.4")
@pytest.mark.skipif(sys.platform == 'win32',
                    reason="does not currently run on windows")
def test_cntk_fastrcnn_eval_evalCorrect(nb):
    # Make sure that the number of detections is more than 0
    detectionCells = [cell for cell in nb.cells
                 if cell.cell_type == 'code' and
                     len(cell.outputs) > 0 and
                     'text' in cell.outputs[0] and
                     re.search('Number of detections: (\d+)',  get_output_stream_from_cell(cell))]
    assert len(detectionCells) == 1
    
    number_of_detections = int(re.search('Number of detections: (\d+)', get_output_stream_from_cell(detectionCells[0])).group(1))
    assert(number_of_detections > 0)


    #Make sure that the last cells was ran successfully
    testCells = [cell for cell in nb.cells
                 if cell.cell_type == 'code' and
                     len(cell.outputs) > 0 and
                     'text' in cell.outputs[0] and
                     re.search('Evaluation result:', get_output_stream_from_cell(cell))]
    assert len(testCells) == 1
