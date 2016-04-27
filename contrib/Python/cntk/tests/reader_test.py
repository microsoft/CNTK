# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *
from ..ops.variables_and_parameters import *
from ..ops import cntk1 as cntk1_ops

from cntk.tests.test_utils import *

# Keeping things short
C = constant
I = input_reader


def test_two_inputs(device_id, precision):
    a = AA([[1, 2]])
    b = AA([[10, 20]])

    expected = a + b

    op_node = I([a], has_dynamic_axis=True) + \
        I([b], has_dynamic_axis=True)

    unittest_helper(op_node, None, [expected], device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)
