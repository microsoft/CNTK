# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the assign operation.
"""

import numpy as np
import pytest
import cntk as C

def test_assign_fw():
    dest = C.constant(shape=(3,4))
    data = C.parameter(shape=(3,4), init=2)
    result = C.assign(dest,data).eval()

    assert np.array_equal(dest.asarray(), data.asarray())
    assert np.array_equal(result, data.asarray())

