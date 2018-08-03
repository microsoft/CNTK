# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np
import pytest

@pytest.fixture(scope="module")
def notebook_path():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_103C_MNIST_MultiLayerPerceptron.ipynb")
    
    return notebook

def test_cntk_103c_mnist_multilayerperceptron_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalErrorByDeviceId = { -1: 1.72, 0: 1.81 }

def test_cntk_103c_mnist_multilayerperceptron_evalCorrect(nb, device_id):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('trainer\.test_minibatch', cell.source)]
    assert len(testCell) == 1
    m = re.match(r"Average test error: (?P<actualEvalError>\d+\.\d+)%\r?$", testCell[0].outputs[0]['text'])
    # TODO tighten tolerances
    assert np.isclose(float(m.group('actualEvalError')), expectedEvalErrorByDeviceId[device_id], atol=0.2)
