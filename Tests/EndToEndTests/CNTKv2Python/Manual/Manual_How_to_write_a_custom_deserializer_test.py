# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "Tutorials"))

from nb_helper import get_output_stream_from_cell

notebook = os.path.join(abs_path, "..", "..", "..", "..", "Manual", "Manual_How_to_write_a_custom_deserializer.ipynb")

def test_cntk_how_to_train_no_errors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedOutput = 'Total number of samples 200000'
def test_csv_correct(nb):
    testCells = [cell for cell in nb.cells
                if cell.cell_type == 'code']
    assert len(testCells) == 6
    text = get_output_stream_from_cell(testCells[5])
    print(text)
    assert re.search(expectedOutput, text)
