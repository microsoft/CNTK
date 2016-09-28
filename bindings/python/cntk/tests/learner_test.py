# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
from ..learner import *
from .. import parameter, input_variable, momentums_per_sample

import pytest

LR_PARAMS = [
        ((0.2,), [0.2]),
        ((0.2,), [0.2, 0.2, 0.2, 0.2]),
        (([0.2,0.4], 5), [0.2]*5+[0.4]*20),
        # TODO does not work yet
        # (([(1,0.2),(2,0.4),(1,0.3)], 5), [0.2]*5+[0.3]*10+[0.5]*30)
        ]
@pytest.mark.parametrize("params, expectation", LR_PARAMS)
def test_learning_rates_per_sample(params, expectation):
    l = learning_rates_per_sample(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def test_learner_init():
    # TODO Test functionality
    i = input_variable(shape=(1,),
            needs_gradient=True,
            name='a')
    w = parameter(shape=(1,))

    res = i*w

    sgd_learner(res.parameters(), lr=0.1)

    momentum_time_constant = 1100
    momentum_per_sample = momentums_per_sample(math.exp(-1.0 / momentum_time_constant))

    momentum_sgd_learner(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    nesterov_learner(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    adagrad_learner(res.parameters(), lr=0.1, need_ave_multiplier=True)

    fsadagrad_learner(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    gamma, inc, dec, max, min = [0.1]*5
    rmsprop_learner(res.parameters(), 0.1, gamma, inc, dec, max, min, True)
