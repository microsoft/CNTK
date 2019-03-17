# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import pytest
import re
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_201B_CIFAR-10_ImageHandsOn.ipynb")
notebook_deviceIdsToRun = [0]
notebook_timeoutSeconds = 900

def test_cntk_201B_cifar_10_imagehandson_noErrors(nb):
    if os.getenv("OS")=="Windows_NT" and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

    metrics = []
    for cell in nb.cells:
        try:
           if cell.cell_type == 'code':
               m = re.search(r'Final Results: .* errs = (?P<metric>\d+\.\d+)%', cell.outputs[0]['text'])
               if m:
                   metrics.append(float(m.group('metric')))
                   break
        except IndexError:
           pass
        except KeyError:
           pass
    # TODO tighten tolerances
    assert np.allclose([43.3], metrics, atol=0.5)
