# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for unit tests
"""

import numpy as np
import pytest

# NumPy's allclose() has 1e08 as the absolute tolerance, which is too strict for
# functions like sigmoid.
TOLERANCE_ABSOLUTE = 1E-06

PRECISION_TO_TYPE = {'float': np.float32, 'double': np.float64}

AA = np.asarray


@pytest.fixture(params=["float", "double"])
def precision(request):
    return request.param
