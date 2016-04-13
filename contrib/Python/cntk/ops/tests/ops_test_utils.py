# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import numpy as np
import pytest
from ...context import get_new_context
from ...graph import *
from ...reader import *

#Keeping things short
C = constant
I = input_reader
AA = np.asarray

@pytest.fixture(params=["float","double"])
def precision(request):
    return request.param

def unittest_helper(root_node, input_reader, expected, device_id = -1, precision="float", 
                    clean_up=True, backward_pass = False, input_node = None):
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        ctx.device_id = device_id
        ctx.precision = precision
        assert not ctx.input_nodes
        result = ctx.eval(root_node, input_reader, backward_pass, input_node)

        assert len(result) == len(expected)
        for res, exp in zip(result, expected):  
            assert np.allclose(res, exp)
            assert res.shape == AA(exp).shape