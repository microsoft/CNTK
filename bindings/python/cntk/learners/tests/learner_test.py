# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
import numpy as np
from .. import *
from cntk import parameter, input

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
    i = input(shape=(1,), needs_gradient=True, name='a')
    w = parameter(shape=(1,))

    res = i * w

    learner = sgd(res.parameters, lr=learning_rate_schedule(0.1, UnitType.sample))
    assert learner.learning_rate() == 0.1
    
    learner.reset_learning_rate(learning_rate_schedule([1,2,3], UnitType.minibatch));
    assert learner.learning_rate() == 1.0

    learner_parameter = learner.parameters
    from cntk.variables import Parameter
    param = learner_parameter[0]
    assert isinstance(param, Parameter)

    unit_gain_value = default_unit_gain_value()
    assert unit_gain_value

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.1, UnitType.sample)
    momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant)
    momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant, unit_gain=unit_gain_value)

    set_default_unit_gain_value(False)
    unit_gain_value = default_unit_gain_value()
    assert not unit_gain_value

    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample)
    nesterov(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)
    nesterov(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    nesterov(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant, unit_gain=unit_gain_value)

    lr_per_sample = learning_rate_schedule([0.1]*3 +[0.2]*2 +[0.3], UnitType.sample)
    adagrad(res.parameters, lr=lr_per_sample, need_ave_multiplier=True)

    set_default_unit_gain_value(True)
    unit_gain_value = default_unit_gain_value()
    assert unit_gain_value

    lr_per_sample = learning_rate_schedule([(3,0.1), (2, 0.2), (1, 0.3)], UnitType.sample)
    fsadagrad(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)
    fsadagrad(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    fsadagrad(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant, unit_gain=unit_gain_value)

    gamma, inc, dec, max, min = [0.1]*5
    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample, 100)
    rmsprop(res.parameters, lr_per_sample, gamma, inc, dec, max, min, True)

    adadelta(res.parameters)

def test_learner_update():
    i = input(shape=(1,), needs_gradient=True, name='a')
    w_init = 1
    w = parameter(shape=(1,), init=w_init)
    res = i * w

    learner = sgd(res.parameters, lr=learning_rate_schedule([0.1]*50 + [0.2]*50, UnitType.sample, 1))
    assert learner.learning_rate() == 0.1
    x = learner.update({w: np.asarray([[2.]], dtype=np.float32)}, 100)
    assert learner.learning_rate() == 0.2
    assert w.value < w_init

    learner.reset_learning_rate(learning_rate_schedule([0.3]*50 + [0.4]*50, UnitType.sample, 1));
    assert learner.learning_rate() == 0.3
    x = learner.update({w: np.asarray([[2.]], dtype=np.float32)}, 100)
    assert learner.learning_rate() == 0.4

def test_training_parameter_schedule():
    training_parameter_schedule(0.01, unit='minibatch')
    training_parameter_schedule(0.01, unit='sample')

    with pytest.raises(ValueError):
        training_parameter_schedule(0.01, unit='not_supported')
    with pytest.raises(ValueError):
        training_parameter_schedule(0.01, unit=5)

def test_sweep_based_schedule(tmpdir, device_id):
    from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
    from cntk import cross_entropy_with_softmax, classification_error, plus, reduce_sum, sequence
    from cntk import Trainer

    input_dim = 69

    ctf_data = '''\
0   |S0 3:1   |S1 3:1 |# <s>
0   |S0 4:1 |# A    |S1 32:1 |# ~AH
0   |S0 5:1 |# B    |S1 36:1 |# ~B
0   |S0 4:1 |# A    |S1 31:1 |# ~AE
0   |S0 7:1 |# D    |S1 38:1 |# ~D
0   |S0 12:1 |# I   |S1 47:1 |# ~IY
0   |S0 1:1 |# </s> |S1 1:1 |# </s>
2   |S0 60:1 |# <s> |S1 3:1 |# <s>
2   |S0 61:1 |# A   |S1 32:1 |# ~AH
'''
    ctf_file = str(tmpdir/'2seqtest.txt')
    with open(ctf_file, 'w') as f:
        f.write(ctf_data)

    mbs = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
    )), randomize=False)

    in1 = sequence.input(shape=(input_dim,))
    labels = sequence.input(shape=(input_dim,))
    p = parameter(shape=(input_dim,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    lr_per_sample = learning_rate_schedule([0.3, 0.2, 0.1, 0.0], UnitType.sample)
    learner = sgd(z.parameters, lr_per_sample)
    trainer = Trainer(z, (ce, errs), [learner])

    input_map = {
        in1       : mbs.streams.features,
        labels : mbs.streams.labels
    }

    # fetch minibatch (first sequence)
    data = mbs.next_minibatch(1, input_map=input_map) 
    trainer.train_minibatch(data)
    assert learner.learning_rate() == 0.3

    # fetch minibatch (second sequence, sweep ends at this point)
    data = mbs.next_minibatch(1, input_map=input_map)
    trainer.train_minibatch(data)
    assert learner.learning_rate() == 0.2

    # fetch minibatch (both sequences -- entire sweep in one go)
    data = mbs.next_minibatch(9, input_map=input_map)
    trainer.train_minibatch(data)
    assert learner.learning_rate() == 0.1

    # fetch minibatch (multiple sweeps)
    data = mbs.next_minibatch(30, input_map=input_map)
    trainer.train_minibatch(data, outputs=[z.output])
    assert learner.learning_rate() == 0.0
