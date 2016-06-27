# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
import cntk as C



REDUCE_SUM_TEST_CASES = [
    ([[10, 0],[20, 1]], 0,        [31],  [[1,1],[1,1]]),
    ([[10, 0],[20, 1]], 1, [[10], [21]], [[1,1],[1,1]]),
    ([[10, 0],[20, 1]], 2, [[30,    1]], [[1,1],[1,1]]),
]
@pytest.mark.parametrize("input_data, axis_data, expected_result, expected_gradient", REDUCE_SUM_TEST_CASES)
def test_op_reduce_sum(input_data, axis_data, expected_result, expected_gradient, device_id, precision):

    a = I([input_data])


    # slice using the operator
    result = C.reduce_sum(a, axis = axis_data)

    unittest_helper(result, None, [[expected_result]], device_id=device_id, 
                precision=precision, clean_up=False, backward_pass=False)