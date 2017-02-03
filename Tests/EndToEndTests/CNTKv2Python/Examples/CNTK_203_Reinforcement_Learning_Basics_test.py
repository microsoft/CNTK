# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_203_Reinforcement_Learning_Basics.ipynb")
notebook_timeoutSeconds = 450

def test_cntk_203_reinforcement_learning_basics_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

def test_cntk_203_reinforcement_learning_basics_tasks_are_solved(nb):
    testCells = [cell for cell in nb.cells
                 if cell.cell_type == 'code' and
                     len(cell.outputs) > 0 and
                     'text' in cell.outputs[0] and
                     re.search('Task solved in[ :]', cell.outputs[0]['text'])]
    assert len(testCells) == 2
