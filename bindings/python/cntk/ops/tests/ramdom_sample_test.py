# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests random sampling related operations
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type
from ...utils import sanitize_dtype_cntk, ones_like, eval
