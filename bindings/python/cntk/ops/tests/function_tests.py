# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the function class.
"""

import numpy as np
import pytest
from ..functions import *
from .. import constant

def test_variable_forwarding():
    op = constant(value=2, shape=(3,4)) + 1
    assert op.shape().dimensions() == (3,4)

