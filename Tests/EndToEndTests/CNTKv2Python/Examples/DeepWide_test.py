# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
deepWide_path = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Recommendsystem", "deepWide")
sys.path.append(deepWide_path)
from deepWide import deepWide

TOLERANCE_ABSOLUTE = 1E-2
def test_deepWide_error():
    error = deepWide()
    expected_error = 0.30578
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)

