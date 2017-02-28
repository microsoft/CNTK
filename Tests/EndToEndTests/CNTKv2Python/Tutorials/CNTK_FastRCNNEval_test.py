# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import sys
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Examples","Image","Detection","FastRCNN", "CNTK_FastRCNN_Eval.ipynb")

# setting a large timeout in case we need to download the Fast-RCNN pretrained model
notebook_timeoutSeconds = 1200

# Skipping test for python 2.7 since Fast-RCNN implementation does not support 2.7 at the moment
@pytest.mark.skipif(sys.version_info < (3,4),
                    reason="requires python 3.4")
def test_cntk_fastrcnn_eval_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]

    assert errors == []

# Skipping test for python 2.7 since Fast-RCNN implementation does not support 2.7 at the moment
@pytest.mark.skipif(sys.version_info < (3,4),
                    reason="requires python 3.4")
def test_cntk_fastrcnn_eval_evalCorrect(nb):
    testCells = [cell for cell in nb.cells
                 if cell.cell_type == 'code' and
                     len(cell.outputs) > 0 and
                     'text' in cell.outputs[0] and
                     re.search('Evaluation result:', cell.outputs[0]['text'])]
    assert len(testCells) == 1
