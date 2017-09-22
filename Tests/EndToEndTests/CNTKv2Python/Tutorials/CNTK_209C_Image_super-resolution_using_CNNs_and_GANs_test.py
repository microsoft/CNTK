# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_209C_Image_Super-resolution_Using_CNNs_and_GANs.ipynb")
datadir = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "DataSets", "BerkeleySegmentationDataset")

# Run this on GPU only
notebook_deviceIdsToRun = [0]

def test_cntk_103b_mnist_logisticregression_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []