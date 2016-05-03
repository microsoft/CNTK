
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for eval() functionality that is used to discover and play with 
operations
"""

import numpy as np
import cntk
from ..ops import plus

import pytest

def test_eval_plus():
    result = cntk.eval(plus([1., 2., 3., 4.], [1., 1., 0., 0.]))
    TOLERANCE_ABSOLUTE = 1E-06    
    assert np.allclose(result, np.asarray([2., 3., 3., 4.]), atol=TOLERANCE_ABSOLUTE)
