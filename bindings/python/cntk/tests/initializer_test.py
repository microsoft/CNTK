# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from ..initializer import *
from .. import parameter


def _check(init, name):
    p = parameter(shape=(10,20,5), init=init)
    val = np.asarray(p)
    assert np.allclose(np.average(val), 0, atol=0.1), name
    assert np.var(val) > 0.01, name

def test_initializer_init(device_id):
    from cntk.utils import cntk_device
    from cntk import cntk_py
    from cntk.device import set_default_device
    cntk_py.always_allow_setting_default_device()
    set_default_device(cntk_device(device_id))

    _check(uniform(scale=10), 'uniform')
    _check(gaussian(output_rank=1, filter_rank=2, scale=10), 'gaussian')
    _check(xavier(output_rank=1, filter_rank=2, scale=10), 'xavier')
    _check(glorot_uniform(output_rank=1, filter_rank=2, scale=10), 'glorot_uniform')
    _check(glorot_normal(output_rank=1, filter_rank=2, scale=10), 'glorot_normal')
    _check(he_uniform(output_rank=1, filter_rank=2, scale=10), 'he_uniform')
    _check(he_normal(output_rank=1, filter_rank=2, scale=10), 'he_normal')
