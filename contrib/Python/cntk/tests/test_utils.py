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


@pytest.fixture(params=["dense", "sparse"])
def left_matrix_type(request):
    return request.param

@pytest.fixture(params=["dense", "sparse"])
def right_matrix_type(request):
    return request.param



def unittest_helper(root_node, input_numpy, expected, device_id=-1, precision="float",
                    clean_up=True, backward_pass=False, input_node=None):
    from cntk.context import get_new_context
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        ctx.device_id = device_id
        ctx.precision = precision
        assert not ctx.input_nodes
        result = ctx.eval(root_node, input_numpy, backward_pass, input_node)
        
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert np.allclose(res, exp, atol=TOLERANCE_ABSOLUTE)
            assert res.shape == AA(exp).shape
