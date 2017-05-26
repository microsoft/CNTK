# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from ..initializer import *
from .. import parameter

def _check_min_max(init, low, high, name):
    p = parameter(shape=(50,20,5), init=init)
    val = p.asarray()
    assert np.max(val) <= high, name
    assert np.min(val) >= low, name

def _check(init, name):
    p = parameter(shape=(50,20,5), init=init)
    val = p.asarray()
    assert np.allclose(np.average(val), 0, atol=0.1), name
    assert np.var(val) > 0.01, name

def test_initializer_init(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    from cntk import cntk_py
    cntk_py.always_allow_setting_default_device()
    from cntk.device import try_set_default_device
    try_set_default_device(cntk_device(device_id))

    _check(uniform(scale=1), 'uniform')
    _check(normal(scale=1, output_rank=1, filter_rank=2), 'normal')
    _check(xavier(scale=10, output_rank=1, filter_rank=2), 'xavier')
    _check(glorot_uniform(scale=10, output_rank=1, filter_rank=2), 'glorot_uniform')
    _check(glorot_normal(scale=10, output_rank=1, filter_rank=2), 'glorot_normal')
    _check(he_uniform(scale=10, output_rank=1, filter_rank=2), 'he_uniform')
    _check(he_normal(scale=10, output_rank=1, filter_rank=2), 'he_normal')
    _check(truncated_normal(stdev=10), 'truncated_gaussian')

    _check_min_max(truncated_normal(stdev=2), -4, 4, 'truncated_gaussian')
