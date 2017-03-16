# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_202_Language_Understanding.ipynb")
# Runs on GPU only, batch normalization training on CPU is not yet implemented.
notebook_deviceIdsToRun = [0]

def test_cntk_202_language_understanding_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

def test_cntk_202_language_understanding_trainerror(nb):
    metrics = []
    for cell in nb.cells:
        try:
           if cell.cell_type == 'code':
               m = re.search('Finished Evaluation.* metric = (?P<metric>\d+\.\d+)%', cell.outputs[0]['text'])
               if m:
                   metrics.append(float(m.group('metric')))
        except IndexError:
           pass
        except KeyError:
           pass
    expectedMetrics = [2.8, 1.9, 2.2, 2.0]
    # TODO tighten tolerances
    assert numpy.allclose(expectedMetrics, metrics, atol=0.2)
