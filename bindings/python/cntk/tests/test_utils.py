# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for unit tests
"""

import numpy as np
import scipy as sp
import pytest
import cntk as C

# NumPy's allclose() has 1e08 as the absolute tolerance, which is too strict for
# functions like sigmoid.
TOLERANCE_ABSOLUTE = 1E-06

PRECISION_TO_TYPE = {'float': np.float32, 'double': np.float64}

AA = np.asarray


@pytest.fixture(params=["float", "double"])
def precision(request):
    return request.param

def _to_dense(val, is_sequence=False):
    if is_sequence:
        x = C.sequence.input(val.shape[2:], is_sparse=True)
    else:
        x = C.input(val.shape[1:], is_sparse=True)

    dense = C.times(x, C.constant(value=np.eye(val.shape[-1], dtype=np.float32)))
    return dense.eval({x : val}, device=val.device)

def _to_csr(data):
    np_data = np.asarray(data, dtype=np.float32)
    data_reshaped = np_data.reshape((-1, np_data.shape[-1]))
    return sp.sparse.csr_matrix(data_reshaped, dtype=np.float32)
