# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_302b_ReasoNet_Evaluation_With_CNN_Dataset.ipynb")
# Runs on GPU only, batch normalization training on CPU is not yet implemented.
notebook_deviceIdsToRun = [0]

def test_cntk_302b_ReasoNet_Evaluation(nb):
  errors = [output for cell in nb.cells if 'outputs' in cell
            for output in cell['outputs'] if output.output_type == "error"]
  print(errors)
  assert errors == []
