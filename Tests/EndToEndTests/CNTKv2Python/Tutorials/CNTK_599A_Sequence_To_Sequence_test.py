# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_599A_Sequence_To_Sequence.ipynb")

def test_cntk_599_sequence_to_sequence_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

expectedEvalError = 90

def test_cntk_599_sequence_to_sequence_trainerror(nb):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Print the training error', cell.source)]
    assert float((testCell[0].outputs[0])['text']) < expectedEvalError
