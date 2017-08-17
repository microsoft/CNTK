# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division, print_function
import numpy as np
import cntk as C
from cntk import parameter

import pytest
import sys

from cntk.logging import ProgressPrinter
from cntk.learners import sgd, learning_rate_schedule, UnitType, universal
from cntk.layers import Dense, Sequential

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
        
LEARNER_LAMBDAS = [
    lambda params: C.adadelta(params),
    lambda params: C.adagrad(params, lr=learning_rate_schedule(1, UnitType.minibatch)),
    lambda params: C.adam(params, lr=learning_rate_schedule(1, UnitType.minibatch), momentum=C.momentum_schedule(0.9)),
    lambda params: C.fsadagrad(params, lr=learning_rate_schedule(1, UnitType.minibatch), momentum=C.momentum_schedule(0.9)),
    lambda params: C.nesterov(params, lr=learning_rate_schedule(1, UnitType.minibatch), momentum=C.momentum_schedule(0.9)),
    lambda params: C.rmsprop(params, lr=learning_rate_schedule(1, UnitType.minibatch), gamma=0.1, inc=3.0, dec=0.1, max=np.inf, min=1e-8),
    lambda params: C.sgd(params, lr=learning_rate_schedule(1, UnitType.minibatch)),
    lambda params: C.momentum_sgd(params, lr=learning_rate_schedule(1, UnitType.minibatch), momentum=C.momentum_schedule(0.9))]

@pytest.mark.parametrize("params, expectation", LR_SCHEDULE_PARAMS)
def test_learning_rate_schedule(params, expectation):
    l = learning_rate_schedule(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def sweep_based_schedule_fails():
    with pytest.raises(Exception):
        learning_rate_schedule([1], unit=UnitType.sample, epoch_size=0)

def test_momentum_schedule():
    m = 2500
    ms = C.momentum_as_time_constant_schedule([m])
    assert ms[0] ==  np.exp(-1.0 / np.asarray(m))

    ms = C.momentum_as_time_constant_schedule(m)
    assert ms[0] ==  np.exp(-1.0 / np.asarray(m))

    mlist = [980, 520]
    msl = C.momentum_as_time_constant_schedule(mlist)
    expected = np.exp(-1.0 / np.asarray(mlist))
    assert all(mi == ei for mi,ei in zip(msl,expected))

@pytest.mark.parametrize("params, expectation", MOMENTUM_SCHEDULE_PARAMS)
def test_momentum_schedule_per_sample(params, expectation):
    l = C.momentum_schedule(*params)
    assert [l[i] for i in range(len(expectation))] == expectation

def test_learner_init():
    i = C.input_variable(shape=(1,), needs_gradient=True, name='a')
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

    unit_gain_value = C.default_unit_gain_value()
    assert unit_gain_value

    momentum_time_constant = C.momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.1, UnitType.sample)
    C.momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant)
    C.momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    C.momentum_sgd(res.parameters, lr_per_sample, momentum_time_constant, unit_gain=unit_gain_value)

    C.set_default_unit_gain_value(False)
    unit_gain_value = C.default_unit_gain_value()
    assert not unit_gain_value

    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample)
    C.nesterov(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)
    C.nesterov(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    C.nesterov(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant, unit_gain=unit_gain_value)

    lr_per_sample = learning_rate_schedule([0.1]*3 +[0.2]*2 +[0.3], UnitType.sample)
    C.adagrad(res.parameters, lr=lr_per_sample, need_ave_multiplier=True)

    C.set_default_unit_gain_value(True)
    unit_gain_value = C.default_unit_gain_value()
    assert unit_gain_value

    lr_per_sample = learning_rate_schedule([(3,0.1), (2, 0.2), (1, 0.3)], UnitType.sample)
    C.fsadagrad(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant)
    C.fsadagrad(res.parameters, lr_per_sample, momentum_time_constant, unit_gain_value)
    C.fsadagrad(res.parameters, lr=lr_per_sample, momentum=momentum_time_constant, unit_gain=unit_gain_value)

    gamma, inc, dec, max, min = [0.5, 1.2, 0.7, 10, 1e-8]
    lr_per_sample = learning_rate_schedule([0.1, 0.2], UnitType.sample, 100)
    C.rmsprop(res.parameters, lr_per_sample, gamma, inc, dec, max, min, True)

    C.set_default_use_mean_gradient_value(False)
    use_mean_gradient_value = C.default_use_mean_gradient_value()
    assert not use_mean_gradient_value

    C.adadelta(res.parameters, lr_per_sample)
    
    C.set_default_use_mean_gradient_value(True)
    use_mean_gradient_value = C.default_use_mean_gradient_value()
    assert use_mean_gradient_value

    C.adadelta(res.parameters, lr_per_sample)

def test_learner_update():
    i = C.input_variable(shape=(1,), needs_gradient=True, name='a')
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


def test_noise_injection_with_checkpointing():
    from cntk import initializer
    shape = (100,100)
    
    w1 = parameter(shape=shape, init=initializer.glorot_uniform(seed=123))
    w2 = parameter(shape=shape, init=initializer.glorot_uniform(seed=123))
    w3 = parameter(shape=shape, init=initializer.glorot_uniform(seed=123))
    
    lr=learning_rate_schedule(0.5, UnitType.sample)
    m=C.momentum_schedule(0.99)

    learner1 = C.momentum_sgd([w1], lr, m, gaussian_noise_injection_std_dev=0.5)
    learner2 = C.momentum_sgd([w2], lr, m, gaussian_noise_injection_std_dev=0.5)
    learner3 = C.momentum_sgd([w3], lr, m, gaussian_noise_injection_std_dev=0.5)

    assert np.allclose(w1.value, w2.value) and np.allclose(w1.value, w3.value)

    for i in range(10):
        checkpoint = learner1.create_checkpoint()

        v =  np.float32(np.random.rand(100,100))
    
        learner1.update({w1: v}, 1)
        learner2.update({w2: v}, 1)
        assert not np.allclose(w1.value, w2.value)

        learner3.restore_from_checkpoint(checkpoint)
        learner3.update({w3: v}, 1)
        assert np.allclose(w1.value, w3.value)

class TestProgressWriter(C.cntk_py.ProgressWriter):

    def __init__(self):
        super(TestProgressWriter, self).__init__(1, 0, 1, 0, sys.maxsize, 0)
        self.log_output = []
        self.__disown__()

    def write(self, key, value):
        self.log_output.append(float(value))

def test_learner_logging():
    from cntk import Trainer
    from cntk.logging import ProgressPrinter
    from cntk import cross_entropy_with_softmax, classification_error

    features = C.input_variable(shape=(1,), needs_gradient=True, name='a')
    w_init = 1
    w = parameter(shape=(1,), init=w_init)
    z = features * w
    labels = C.input_variable(shape=(1,), name='b')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    writer = TestProgressWriter();
    lr_values = [0.3, 0.2, 0.1, 0]
    m_values = [0.6, 0.7, 0.8]
    learner = C.momentum_sgd(z.parameters,
                  learning_rate_schedule(lr_values, UnitType.sample, 1),
                  C.momentum_schedule(m_values, 1))
    trainer = Trainer(z, (ce, errs), [learner], writer)

    for i in range(10):
        trainer.train_minibatch({features: [[2.]], labels: [[1.]]})
    
    assert len(writer.log_output) == len(lr_values + m_values)

    values = [j for i in zip(lr_values,m_values) for j in i] + [0]

    for i in range(len(values)):
        assert (values[i] == writer.log_output[i])

def test_training_parameter_schedule():
    C.training_parameter_schedule(0.01, unit='minibatch')
    C.training_parameter_schedule(0.01, unit='sample')

    with pytest.raises(ValueError):
        C.training_parameter_schedule(0.01, unit='not_supported')
    with pytest.raises(ValueError):
        C.training_parameter_schedule(0.01, unit=5)

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

    in1 = sequence.input_variable(shape=(input_dim,))
    labels = sequence.input_variable(shape=(input_dim,))
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


def generate_random_data(sample_size, feature_dim, num_classes):
     # Create synthetic data using NumPy.
     Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

     # Make sure that the data is separable
     X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
     X = X.astype(np.float32)
     # converting class 0 into the vector "1 0 0",
     # class 1 into vector "0 1 0", ...
     class_ind = [Y == class_number for class_number in range(num_classes)]
     Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
     return X, Y


def test_learner_empy_parameters_list():
    lr_per_sample = learning_rate_schedule(0.1, UnitType.sample)
    with pytest.raises(ValueError):
        learner = C.sgd([], lr_per_sample)


def ffnet(learner, trainer=None):
    inputs = 5
    outputs = 3
    layers = 2
    hidden_dimension = 3

    if trainer is None:
        # input variables denoting the features and label data
        features = C.input_variable((inputs), np.float32)
        label = C.input_variable((outputs), np.float32)

        # Instantiate the feedforward classification model
        my_model = Sequential ([
                        Dense(hidden_dimension, activation=C.sigmoid, init=C.glorot_uniform(seed=98052)),
                        Dense(outputs, init=C.glorot_uniform(seed=98052))])
        z = my_model(features)

        ce = C.cross_entropy_with_softmax(z, label)
        pe = C.classification_error(z, label)

        # Instantiate the trainer object to drive the model training
        progress_printer = ProgressPrinter(0)
        trainer = C.Trainer(z, (ce, pe), [learner(z.parameters)], [progress_printer])
    else:
        features = trainer.loss_function.arguments[0]
        label = trainer.loss_function.arguments[1]

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 100

    aggregate_loss = 0.0
    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        trainer.train_minibatch({features : train_features, label : labels})
        sample_count = trainer.previous_minibatch_sample_count
        aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

    last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

    test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return last_avg_error, avg_error, trainer

def test_sgd_with_noise():
    # Runs a network where the number of parameters is odd
    # in some layers. This tests that cuRand library will not
    # complain about generating an odd number of random values
    np.random.seed(98052)
    learner = lambda params: sgd(params, lr=learning_rate_schedule(0.125, UnitType.minibatch), gaussian_noise_injection_std_dev=0.01)
    ffnet(learner)
    # We just verify that we did not crash
    assert(True)

def test_universal():
    np.random.seed(98052)
    builtin_sgd = lambda params: sgd(params, lr=learning_rate_schedule(0.125, UnitType.minibatch))
    builtin_last_avg_error, builtin_avg_error, _ = ffnet(builtin_sgd)
    np.random.seed(98052)
    my_sgd = lambda ps, gs: C.combine([C.assign(p, p - 0.125/25 * g) for p, g in zip(ps, gs)])
    universal_sgd = lambda params: universal(my_sgd, params)
    my_last_avg_error, my_avg_error, _ = ffnet(universal_sgd)
    assert np.allclose(my_last_avg_error, builtin_last_avg_error)
    assert np.allclose(my_avg_error, builtin_avg_error)

def test_0d_1d_parameter_set_value():
    x = C.input_variable(2)
    w_0d = C.parameter(())
    op = x + w_0d
    w_0d_grad = op.grad({x : np.asarray([1, 2], dtype=np.float32)}, wrt=[w_0d], as_numpy=False)
    w_0d.value = w_0d_grad.data
    assert w_0d.value == 2.

    w_1d = C.parameter(shape=2)
    op = x + w_1d
    w_1d_grad = op.grad({x : np.asarray([1, 2], dtype=np.float32)}, wrt=[w_1d], as_numpy=False)
    w_1d.value = w_1d_grad.data
    assert np.array_equal(w_1d.value, [1., 1.])

@pytest.mark.parametrize("learner", LEARNER_LAMBDAS)
def test_restore_from_checkpoint(tmpdir, learner):
    np.random.seed(0)
    last_avg_err1, avg_err1, trainer1 = ffnet(learner)
    np.random.seed(0)
    last_avg_err2, avg_err2, trainer2 = ffnet(learner)
    
    assert np.allclose(last_avg_err1, last_avg_err2)
    assert np.allclose(avg_err1, avg_err2)

    # create a checkpoint for trainer and continue training
    checkpoint_filename = str(tmpdir.join('checkpoint'))
    trainer2.save_checkpoint(checkpoint_filename)
    np.random.seed(1)
    last_avg_err1, avg_err1, _ = ffnet(None, trainer1)
    np.random.seed(1)
    last_avg_err2, avg_err2, _ = ffnet(None, trainer2)
    assert np.allclose(last_avg_err1, last_avg_err2)
    assert np.allclose(avg_err1, avg_err2)

    # restore from the checkpoint, make sure results match the one without checkpointing
    trainer2.restore_from_checkpoint(checkpoint_filename)
    np.random.seed(1)
    last_avg_err2, avg_err2, _ = ffnet(None, trainer2)
    assert np.allclose(last_avg_err1, last_avg_err2)
    assert np.allclose(avg_err1, avg_err2)
