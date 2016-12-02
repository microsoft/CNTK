# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..learner import *
from .. import parameter, input_variable

import pytest

LR_SCHEDULE_PARAMS = [
        ((0.2, UnitType.sample), [0.2]),
        ((0.2, UnitType.sample), [0.2, 0.2, 0.2, 0.2]),
        (([0.2,0.4], UnitType.sample, 5), [0.2]*5+[0.4]*20),
        (([(3,0.2),(2,0.4),(1,0.8)], UnitType.sample, 5), [0.2]*15+[0.4]*10+[0.8]*20),
        ]

MOMENTUM_SCHEDULE_PARAMS = [
        ((0.2,), [0.2]),
        ((0.2,), [0.2, 0.2, 0.2, 0.2]),
        (([0.2,0.4], 5), [0.2]*5+[0.4]*20),
        (([(3,0.2),(2,0.4),(1,0.8)], 5), [0.2]*15+[0.4]*10+[0.8]*20),
        ]

@pytest.mark.parametrize("params, expectation", LR_SCHEDULE_PARAMS)
def test_learning_rate_schedule(params, expectation):
    l = learning_rate_schedule(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def sweep_based_schedule_fails():
    with pytest.raises(Exception):
        learning_rate_schedule([1], unit=UnitType.sample, epoch_size=0)

def test_momentum_schedule():
    m = 2500
    ms = momentum_as_time_constant_schedule([m])
    assert ms[0] ==  np.exp(-1.0 / np.asarray(m))

    ms = momentum_as_time_constant_schedule(m)
    assert ms[0] ==  np.exp(-1.0 / np.asarray(m))

    mlist = [980, 520]
    msl = momentum_as_time_constant_schedule(mlist)
    expected = np.exp(-1.0 / np.asarray(mlist))
    assert all(mi == ei for mi,ei in zip(msl,expected))

@pytest.mark.parametrize("params, expectation", MOMENTUM_SCHEDULE_PARAMS)
def test_momentum_schedule_per_sample(params, expectation):
    l = momentum_schedule(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def test_learner_init():
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='a')
    w = parameter(shape=(1,))

    res = i * w

    learner = sgd(res.parameters, lr=learning_rate_schedule(0.1, UnitType.sample))
    assert learner.learning_rate() == 0.1
    
    learner.reset_learning_rate(learning_rate_schedule([1,2,3], UnitType.minibatch));
    assert learner.learning_rate() == 1.0

    learner_parameter = learner.parameters
    from ..ops.variables import Parameter
    param = learner_parameter[0]
    assert isinstance(param, Parameter)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.1, UnitType.sample)
    momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant)

    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample)
    nesterov(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)

    lr_per_sample = learning_rate_schedule([0.1]*3 +[0.2]*2 +[0.3], UnitType.sample)
    adagrad(res.parameters, lr=lr_per_sample, need_ave_multiplier=True)

    lr_per_sample = learning_rate_schedule([(3,0.1), (2, 0.2), (1, 0.3)], UnitType.sample)
    adam_sgd(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)

    gamma, inc, dec, max, min = [0.1]*5
    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample, 100)
    rmsprop(res.parameters, lr_per_sample, gamma, inc, dec, max, min, True)

def test_learner_update():
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='a')
    w_init = 1
    w = parameter(shape=(1,), init=w_init)
    res = i * w

    learner = sgd(res.parameters, lr=learning_rate_schedule([0.1]*50 + [0.2]*50, UnitType.sample))
    assert learner.learning_rate() == 0.1
    x = learner.update({w: np.asarray([[2.]], dtype=np.float32)}, 100)
    assert learner.learning_rate() == 0.2
    assert w.value < w_init

def test_training_parameter_schedule():
    training_parameter_schedule(0.01, unit='minibatch')
    training_parameter_schedule(0.01, unit='sample')

    with pytest.raises(ValueError):
        training_parameter_schedule(0.01, unit='not_supported')
    with pytest.raises(ValueError):
        training_parameter_schedule(0.01, unit=5)
