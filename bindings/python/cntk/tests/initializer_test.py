# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from ..initializer import *
from .. import parameter, input_variable, momentums_per_sample

def _check(init, name):
    p = parameter(shape=(10,20,5), initializer=init)
    assert np.allclose(np.average(p.value().to_numpy()), 0, atol=0.1), name
    assert np.var(p.value().to_numpy()) > 0.01, name

def _disabled_test_initializer_init():
    _check(uniform(scale=10), 'uniform')
    _check(gaussian(output_rank=1, filter_rank=2, scale=10), 'gaussian')
    _check(xavier(output_rank=1, filter_rank=2, scale=10), 'xavier')
    _check(glorot_uniform(output_rank=1, filter_rank=2, scale=10), 'glorot_uniform')
    _check(glorot_normal(output_rank=1, filter_rank=2, scale=10), 'glorot_normal')
    _check(he_uniform(output_rank=1, filter_rank=2, scale=10), 'he_uniform')
    _check(he_normal(output_rank=1, filter_rank=2, scale=10), 'he_normal')
