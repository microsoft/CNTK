# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..learner import *
from .. import parameter, input_variable

import pytest

SCHEDULE_PARAMS = [
        ((0.2,), [0.2]),
        ((0.2,), [0.2, 0.2, 0.2, 0.2]),
        (([0.2,0.4], 5), [0.2]*5+[0.4]*20),
        ]
@pytest.mark.parametrize("params, expectation", SCHEDULE_PARAMS)
def test_learning_rates_per_sample(params, expectation):
    l = learning_rates_per_sample(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

@pytest.mark.parametrize("params, expectation", SCHEDULE_PARAMS)
def test_momentums_per_sample(params, expectation):
    l = momentums_per_sample(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def test_learner_init():
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='a')
    w = parameter(shape=(1,))

    res = i * w

    learner = sgd(res.parameters(), lr=0.1)

    learner_parameter = learner.parameters()
    from ..ops.variables import Parameter
    param = learner_parameter.pop()
    assert isinstance(param, Parameter)

    momentum_time_constant = 1100
    momentum_per_sample = momentums_per_sample(
        np.exp(-1.0 / momentum_time_constant))

    momentum_sgd(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    nesterov(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    adagrad(res.parameters(), lr=0.1, need_ave_multiplier=True)

    fsadagrad(res.parameters(), lr=0.1, momentums=momentum_per_sample)

    gamma, inc, dec, max, min = [0.1]*5
    rmsprop(res.parameters(), 0.1, gamma, inc, dec, max, min, True)

def test_learner_update():
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='a')
    w_init = 1
    w = parameter(shape=(1,), init=w_init)
    res = i * w

    learner = sgd(res.parameters(), lr=0.1)
    x = learner.update({w: np.asarray([[2.]], dtype=np.float32)}, 1)
    assert w.value().to_numpy() < w_init

