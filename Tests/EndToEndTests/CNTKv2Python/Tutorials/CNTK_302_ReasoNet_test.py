# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_302_ReasoNet.ipynb")
# Runs on GPU only, batch normalization training on CPU is not yet implemented.
notebook_deviceIdsToRun = [0]

def test_cntk_302_ReasoNet_Training(nb):
  if sys.version_info >= (3,):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []
    metrics = []
    train_acc = nb.cells[37].outputs[0]['text']
    m = re.search('^Evaluation Acc: (?P<metric>\d+\.\d+)', train_acc, re.M)
    metrics += [float(m.group('metric'))]
    eval_acc = nb.cells[42].outputs[0]['text']
    m = re.search('^Evaluation Acc: (?P<metric>\d+\.\d+)', eval_acc, re.M)
    metrics += [float(m.group('metric'))]
    expectedMetrics = [0.336, 0.32]
    # TODO tighten tolerances
    assert numpy.allclose(expectedMetrics, metrics, atol=0.01)
