# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os

abs_path = os.path.dirname(os.path.abspath(__file__))

notebook = os.path.join(abs_path, "..", "..", "..", "..", "Manual", "Manual_How_to_use_and_evaluate_different_learners.ipynb")

# Note: Given this is a manual for learners, we check only for functional correctness of API.


def test_manual_use_evaluate_learners_no_errors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []
